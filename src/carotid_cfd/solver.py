"""
carotid_cfd.solver
==================
Time-stepping loop using fem.petsc.NonlinearProblem — dolfinx API.

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

from carotid_cfd.config   import SimulationConfig
from carotid_cfd.geometry import load_mesh
from carotid_cfd.spaces   import build_spaces
from carotid_cfd.boundary import BoundaryConditions
from carotid_cfd.weakform import NavierStokesForm
from carotid_cfd.postproc import FlowMonitor


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


    cell_name = mesh.topology.cell_name()
    gdim = mesh.geometry.dim

    P1_vec = basix.ufl.element(
        "Lagrange",
        cell_name,
        1,
        shape=(gdim,)
    )

    # P1 = basix.ufl.element(
    #     "Lagrange",
    #     cell_name,
    #     1
    # )

    V_out = fem.functionspace(mesh, P1_vec)

    # Pressure output space (P1 scalar)
    Q_out = fem.functionspace(mesh, ("CG", 1))

    # ── Previous-step velocity (t^n) in the collapsed velocity space ──────────
    # Lives in V (not W) — same pattern as u_previous in the reference example.
    # Zero-initialised = zero initial condition for the flow.
    u_previous = fem.Function(V, name="u_previous")

    # ── Functions for output — pre-allocated, reused every write step ─────────
    # Avoids allocating a new Function inside the time loop.
    u_out = fem.Function(V_out, name="velocity")
    p_out = fem.Function(Q_out, name="pressure")

    # ── Boundary conditions ───────────────────────────────────────────────────
    bcs = BoundaryConditions(spaces, mesh, facet_tags, cfg)

    # ── XDMF output files ─────────────────────────────────────────────────────
    xdmf_u = XDMFFile(comm, str(cfg.output_dir / "velocity.xdmf"),  "w")
    xdmf_p = XDMFFile(comm, str(cfg.output_dir / "pressure.xdmf"),  "w")
    xdmf_u.write_mesh(mesh)
    xdmf_p.write_mesh(mesh)

    # ── Flow monitor ──────────────────────────────────────────────────────────
    monitor = FlowMonitor(cfg, mesh, facet_tags)

    # ── PETSc solver options ──────────────────────────────────────────────────
    # Direct LU (MUMPS) is robust and requires no tuning.
    # For very large meshes (>1M DOFs), switch to iterative:
    #   "ksp_type": "gmres", "pc_type": "hypre"
    petsc_options = {
        "snes_type":                   "newtonls",
        "snes_rtol":                    cfg.solver.newton_rel_tol,
        "snes_atol":                    cfg.solver.newton_abs_tol,
        "snes_max_it":                  cfg.solver.newton_max_iter,
        "ksp_type":                    cfg.solver.ksp_type,
        "pc_type":                     cfg.solver.pc_type,
        # Uncomment for direct LU (robust, no convergence issues):
        # "ksp_type":                  "preonly",
        # "pc_type":                   "lu",
        # "pc_factor_mat_solver_type": "mumps",
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
            petsc_options=petsc_options, petsc_options_prefix="healthy_"
        )
        problem.solve()

        # ── Advance solution: u_previous <- velocity part of u_p ─────────────
        # WV_map maps the velocity DOF indices in W to indices in V.
        # This is the reference-example pattern — no allocations.
        u_previous.x.array[:] = ns.u_p.x.array[WV_map]

        # ── Monitor ───────────────────────────────────────────────────────────
        rec = monitor.compute(t, ns.u_p)

        # ── Output ────────────────────────────────────────────────────────────
        if step % cfg.solver.output_interval == 0:
            _print_step(t, step, rec)

            # u_out.x.array[:] = ns.u_p.x.array[WV_map]
            # p_out.x.array[:] = ns.u_p.x.array[WQ_map]

            u_out.interpolate(ns.u_p.sub(0))
            p_out.interpolate(ns.u_p.sub(1))

            xdmf_u.write_function(u_out, t)
            xdmf_p.write_function(p_out, t)

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
