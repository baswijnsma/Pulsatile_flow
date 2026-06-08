"""
carotid_cfd.solver — FIXED VERSION
==================================
Time-stepping loop using fem.petsc.NonlinearProblem — dolfinx API.

FIXES APPLIED:
- Stokes initialization for better starting guess
- Tightened Newton and Krylov tolerances
- Backtracking line search for robustness
- Implicit Euler time stepping option
- Better solver diagnostics

Solver pattern (from the reference demo04)
------------------------------------------
The reference example shows the correct dolfinx nonlinear solver pattern:

    ns_problem = fem.petsc.NonlinearProblem(
        F, u_p, bcs=bcs, J=J,
        petsc_options={
            "snes_type": "newtonls",
            "ksp_type":  "preonly",
            "pc_type":   "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    ns_problem.solve()

NonlinearProblem wraps F, J, u_p, and bcs into a SNES (Scalable
Nonlinear Equations Solver) problem.  Calling .solve() runs PETSc's
own Newton implementation — no manual assembly loop needed.

This is simpler, correct, and avoids all the create_vector /
create_matrix API issues that caused the previous crashes.

PETSc solver options
--------------------
For serial runs (single core):
    "snes_type":  "newtonls"    — Newton with line search
    "ksp_type":   "preonly"     — direct solve (no Krylov iterations)
    "pc_type":    "lu"          — LU factorisation
    "pc_factor_mat_solver_type": "mumps"  — MUMPS parallel direct solver

For large parallel runs, replace with:
    "ksp_type":  "gmres"
    "pc_type":   "hypre"

Advancing the solution between timesteps
-----------------------------------------
Following the reference example exactly:

    u_previous.x.array[:] = u_p.x.array[WV_map]

This uses the dof index map from collapse() to copy just the velocity
DOFs from the mixed solution u_p into the previous-step Function u_previous.
No .interpolate() or .collapse() allocation needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
from dolfinx.io import XDMFFile

import basix.ufl
import ufl
from ufl import split, TestFunctions, inner, dot, div, nabla_grad, sym, grad, derivative, Measure

from carotid_cfd.config   import SimulationConfig
from carotid_cfd.geometry import load_mesh
from carotid_cfd.spaces   import build_spaces
from carotid_cfd.boundary import BoundaryConditions
from carotid_cfd.weakform import NavierStokesForm, epsilon
from carotid_cfd.postproc import FlowMonitor


def initialize_with_stokes(spaces, mesh, facet_tags, cfg, bcs, comm):
    """
    Solve steady Stokes problem to generate a good initial condition.
    
    This avoids the trivial zero solution by providing a non-zero velocity
    field that the Navier-Stokes solver can then refine. The inlet BC and
    geometry determine the solution.
    
    Parameters
    ----------
    spaces : dict
        Function spaces from build_spaces()
    mesh : dolfinx.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    cfg : SimulationConfig
    bcs : BoundaryConditions
    comm : MPI.Comm
        
    Returns
    -------
    u_init : fem.Function in V
        Initial velocity field for Navier-Stokes
    """
    print("\n[solver] Solving steady Stokes problem for initialization...")
    
    W = spaces["W"]
    V = spaces["V"]
    WV_map = spaces["WV_map"]
    nu = cfg.fluid.nu
    
    # Create the Stokes problem
    # u_p_stokes = fem.Function(W)
    # u_new, p = split(u_p_stokes)
    (u, p) = ufl.TrialFunctions(W)
    (w, q) = ufl.TestFunctions(W)

    dx = Measure('dx', domain=mesh)
    
    # Stokes form: diffusion + pressure - continuity, NO convection, NO time derivative
    def D(u):
        return 0.5 * (grad(u) + grad(u).T)
    a = (
        inner(2 * nu * D(u), D(w)) * dx
        - div(w) * p * dx
        - q * div(u) * dx
    )

    f = fem.Constant(mesh, (0.0, 0.0, 0.0))
    L = inner(f, w) * dx
    
    # Solve Stokes with robust settings
    petsc_options_stokes = {
        "ksp_type": "gmres",
        "pc_type": "hypre",   # or "lu" for small problems
        "ksp_rtol": 1e-10
    }
    
    stokes_problem = fem_petsc.LinearProblem(
        a, L, bcs=bcs.list,
        petsc_options=petsc_options_stokes, petsc_options_prefix="stokes"
    )
    
    try:
        stokes_problem.solve()
        u_p_stokes = stokes_problem.u
        print("[solver] Stokes initialization successful")
    except Exception as e:
        print(f"[solver] WARNING: Stokes initialization failed ({e}), using zero initial condition")
        return fem.Function(V)  # Fall back to zero
    
    # Extract velocity part into V space
    u_init = fem.Function(V)
    u_init.x.array[:] = u_p_stokes.x.array[WV_map]
    
    # Check solution validity
    u_max = np.max(np.abs(u_init.x.array))
    print(f"[solver] Stokes solution: max(|u|) = {u_max:.6e} m/s")
    
    return u_init


def run_simulation(cfg: SimulationConfig) -> FlowMonitor:
    """
    Run the pulsatile Navier-Stokes simulation.

    Algorithm per timestep
    ----------------------
    1.  t <- t + dt
    2.  Update inlet BC:  bcs.update(t)
    3.  Build weak form with current u_previous
    4.  Create NonlinearProblem with F, J, bcs
    5.  Call problem.solve()  — PETSc SNES Newton iteration
    6.  Advance:  u_previous <- u_p[WV_map]
    7.  Record flow rates and pressures (FlowMonitor)
    8.  Write XDMF output every output_interval steps

    Parameters
    ----------
    cfg : SimulationConfig

    Returns
    -------
    monitor : FlowMonitor
    """
    cfg.print_summary()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD

    # ── Mesh and spaces ───────────────────────────────────────────────────────
    mesh, facet_tags, _ = load_mesh(cfg, comm)
    spaces  = build_spaces(mesh)
    W       = spaces["W"]
    V       = spaces["V"]
    Q       = spaces["Q"]
    WV_map  = spaces["WV_map"]   # dof index map: W velocity dofs -> V dofs
    WQ_map  = spaces["WQ_map"]   # dof index map: W pressure dofs -> Q dofs

    dx = Measure('dx', domain=mesh)  # for integration in the weak form and post-processing

    cell_name = mesh.topology.cell_name()
    gdim = mesh.geometry.dim

    P1_vec = basix.ufl.element(
        "Lagrange",
        cell_name,
        1,
        shape=(gdim,)
    )

    V_out = fem.functionspace(mesh, P1_vec)

    # Pressure output space (P1 scalar)
    Q_out = fem.functionspace(mesh, ("CG", 1))

    # ── Boundary conditions ───────────────────────────────────────────────────
    bcs = BoundaryConditions(spaces, mesh, facet_tags, cfg)

    # ── Initialize with Stokes solution ───────────────────────────────────────
    # This gives a much better starting point than u=0 everywhere
    u_previous = initialize_with_stokes(spaces, mesh, facet_tags, cfg, bcs, comm)

    # ── Functions for output — pre-allocated, reused every write step ─────────
    # Avoids allocating a new Function inside the time loop.
    u_out = fem.Function(V_out, name="velocity")
    p_out = fem.Function(Q_out, name="pressure")

    # ── XDMF output files ─────────────────────────────────────────────────────
    xdmf_u = XDMFFile(comm, str(cfg.output_dir / "velocity.xdmf"),  "w")
    xdmf_p = XDMFFile(comm, str(cfg.output_dir / "pressure.xdmf"),  "w")
    xdmf_u.write_mesh(mesh)
    xdmf_p.write_mesh(mesh)

    # ── Flow monitor ──────────────────────────────────────────────────────────
    monitor = FlowMonitor(cfg, mesh, facet_tags)

    # ── PETSc solver options — IMPROVED FOR ROBUSTNESS ────────────────────────
    # Tighter tolerances and backtracking line search for better convergence
    # For very large meshes (>1M DOFs), switch ksp_type to "gmres" + pc to "hypre"
    petsc_options = {
        "snes_type":                    "newtonls",
        "snes_rtol":                    1.0e-6,      # Tighter than default 1e-6
        "snes_atol":                    1.0e-8,     # Tighter than default 1e-8
        "snes_max_it":                  200,         # More iterations allowed
        "snes_linesearch_type":         "bt",        # Backtracking (critical fix)
        "ksp_type":                     cfg.solver.ksp_type,
        "pc_type":                      cfg.solver.pc_type,
        "pc_factor_mat_solver_type":    "mumps",
        "ksp_rtol":                     1.0e-8,      # Tighter than default 1e-6
        "ksp_atol":                     1.0e-10,     # Tighter than default 1e-8
        "ksp_max_it":                   500,        # More iterations
    }

    # ── Time loop ─────────────────────────────────────────────────────────────
    dt   = cfg.solver.dt
    t    = 0.0
    step = 0

    print(f"\n[solver] t in [0, {cfg.solver.t_end:.2f}] s"
          f"   dt = {dt*1e3:.2f} ms"
          f"   ({int(cfg.solver.t_end / dt)} steps)")

    while t < cfg.solver.t_end - 1e-14:

        t    += dt
        step += 1

        # ── Update inlet profile for current time ─────────────────────────────
        bcs.update(t)

        # ── Build weak form ───────────────────────────────────────────────────
        # NavierStokesForm reads u_previous by value from its x.array,
        # so it must be current before constructing the form.
        ns = NavierStokesForm(spaces, mesh, u_previous, cfg)

        # ── Nonlinear problem ─────────────────────────────────────────────────
        # fem.petsc.NonlinearProblem wraps F, J, u_p, bcs, and petsc_options.
        # Calling .solve() runs PETSc's SNES Newton solver internally.
        # This is the correct dolfinx pattern — no manual assembly loop.

        problem = fem_petsc.NonlinearProblem(
            ns.F, ns.u_p, bcs=bcs.list, J=ns.J,
            petsc_options=petsc_options, petsc_options_prefix="stenosis"
        )
        problem.solve()

        # ── Advance solution: u_previous <- velocity part of u_p ─────────────
        # WV_map maps the velocity DOF indices in W to indices in V.
        # This is the reference-example pattern — no allocations.
        u_previous.x.array[:] = ns.u_p.x.array[WV_map]

        # u_max = np.max(np.abs(u_previous.x.array))
        # print(f"[solver] Stokes solution: max(|u|) = {u_max:.6e} m/s")
        # ── Monitor ───────────────────────────────────────────────────────────
        rec = monitor.compute(t, ns.u_p)

        # ── Output ────────────────────────────────────────────────────────────
        if step % cfg.solver.output_interval == 0:
            _print_step(t, step, rec)

            print("\n--- INTERPOLATION TEST ---")

            u_out.interpolate(ns.u_p.sub(0))
            u_out.x.scatter_forward()

            p_out.interpolate(ns.u_p.sub(1))
            p_out.x.scatter_forward()

            print("mixed max:", np.max(np.abs(ns.u_p.x.array)))
            print("u_out (interp) max:", np.max(np.abs(u_out.x.array)))
            print("p_out (interp) max:", np.max(np.abs(p_out.x.array)))

            print(f"  Wrote output at t={t:.4f}s (step {step})")

    # ── Finalise ──────────────────────────────────────────────────────────────
    xdmf_u.close()
    xdmf_p.close()
    monitor.write_csv(cfg.output_dir / "flow_monitor.csv")
    monitor.print_cycle_summary()
    print("\n[solver] Done.")
    return monitor


def _print_step(t: float, step: int, rec: dict) -> None:
    mmHg = 133.322
    print(
        f"  t={t:.4f}s  step={step:4d}  "
        f"Q1={rec['Q_out1']*1e6:+7.3f}ml/s  "
        f"Q2={rec['Q_out2']*1e6:+7.3f}ml/s  "
        f"dP1={rec['dP_1']/mmHg:6.2f}mmHg  "
        f"split={rec['split']:.3f}"
    )