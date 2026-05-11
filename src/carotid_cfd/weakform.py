"""
carotid_cfd.weakform
====================
Variational formulation — matches the weak form in the report exactly.

Strong form
-----------
    nabla . v = 0                                         in Omega
    d_t v + (v . nabla) v = -(1/rho) nabla p + nu Dv    in Omega

Boundary conditions
-------------------
    v = v_D(t) = phi(x,y) (u_mean + A sin(omega t)) e_1   on Gamma_in
    v = 0                                                   on Gamma_wall
    sigma . n = 0          (natural — no integral needed)   on Gamma_out

Weak form (time-discrete, as written in the report)
----------------------------------------------------
Find (v^{n+1}, p^{n+1}) in V x Q such that for all (w, q) in V_0 x Q:

    int_Omega  w . (v^{n+1} - v^n) / dt  dx
  + int_Omega  w . (v^{n+1} . nabla v^{n+1})  dx        <- fully nonlinear
  + int_Omega  2 nu  epsilon(v^{n+1}) : epsilon(w)  dx
  - int_Omega  p^{n+1}  nabla . w  dx
  + int_Omega  q  nabla . v^{n+1}  dx
  = 0

where epsilon(v) = sym(nabla v) = (nabla v + (nabla v)^T) / 2.

Note: the outlet condition sigma.n = 0 is NATURAL — it appears
automatically when you integrate by parts and impose no Dirichlet BC
on the outlet. No boundary integral is added explicitly.

Correspondence with the reference example (demo04)
---------------------------------------------------
    demo04 uses kinematic viscosity nu and epsilon(u):epsilon(v).
    We follow the same pattern.  The previous implementation used
    dynamic viscosity mu and sigma_fluid : sym(grad(v)) — equivalent
    mathematically but inconsistent with your report notation.

    demo04 pattern:
        u = (u_new + u_previous) / 2          Crank-Nicolson
        dudt = (u_new - u_previous) / dt
        F = dot(dudt, v)*dx
          + inner(dot(u, nabla_grad(u)), v)*dx
          + inner(2*nu*epsilon(u), epsilon(v))*dx
          - dot(div(v), p)*dx
          + dot(q, div(u))*dx

    We use the same structure but set theta=1 (implicit Euler) as
    default for robustness, with theta=0.5 (Crank-Nicolson) available.
    The convection term is fully nonlinear in v^{n+1} — the Jacobian
    (derived automatically by UFL) handles the linearisation inside
    the Newton solver.
"""

from __future__ import annotations

from dolfinx.mesh import Mesh
import dolfinx.fem as fem
import ufl
from ufl import (
    split, TestFunctions,
    inner, dot, div, nabla_grad, sym, grad,
    dx, derivative,
)

from carotid_cfd.config import SimulationConfig


def epsilon(v):
    """Symmetric strain-rate tensor: epsilon(v) = sym(nabla v)."""
    return sym(nabla_grad(v))


class NavierStokesForm:
    """
    Weak form of the incompressible Navier-Stokes equations.

    Matches the notation in the report and the structure of demo04.

    Parameters
    ----------
    spaces     : dict from build_spaces()
    mesh       : dolfinx Mesh
    u_previous : fem.Function in V (velocity at t^n, collapsed space)
    cfg        : SimulationConfig

    Attributes
    ----------
    u_p        : fem.Function in W — the coupled unknown (v^{n+1}, p^{n+1})
    F          : UFL Form — nonlinear residual  F(u_p; w, q) = 0
    J          : UFL Form — Jacobian dF/du_p (automatic differentiation)
    """

    def __init__(
        self,
        spaces:     dict,
        mesh:       Mesh,
        u_previous: fem.Function,   # velocity at t^n  in collapsed V
        cfg:        SimulationConfig,
    ):
        W   = spaces["W"]
        nu  = cfg.fluid.nu      # kinematic viscosity  [m^2/s]  = mu/rho
        dt  = cfg.solver.dt
        th  = cfg.solver.theta  # theta=0.5: Crank-Nicolson, theta=1: implicit Euler

        # ── Mixed solution function: the Newton solver iterates on this ───────
        self.u_p = fem.Function(W)
        self.u_p.name = "u_p"

        # ── Split into velocity and pressure components ────────────────────────
        # ufl.split gives symbolic expressions, not copies.
        # u_new and p refer to components of u_p.
        u_new, p = split(self.u_p)

        # ── Test functions ────────────────────────────────────────────────────
        w, q = TestFunctions(W)

        # ── Crank-Nicolson weighted velocity ──────────────────────────────────
        # theta=0.5: average of new and old  (2nd order, unconditionally stable)
        # theta=1.0: fully implicit / backward Euler  (1st order, more robust)
        u = th * u_new + (1.0 - th) * u_previous

        # ── Weak form — term by term, matching the report exactly ─────────────

        # Temporal term:  int w . (v^{n+1} - v^n) / dt  dx
        F_time = dot((u_new - u_previous) / dt, w) * dx

        # Convection:  int w . (u . nabla u)  dx
        # Fully nonlinear in u_new — Newton handles the linearisation.
        # nabla_grad(u)[i,j] = d u_i / d x_j  (correct for vector fields)
        F_conv = inner(dot(u, nabla_grad(u)), w) * dx

        # Diffusion:  int 2 nu  epsilon(u) : epsilon(w)  dx
        F_diff = inner(2 * nu * epsilon(u), epsilon(w)) * dx

        # Pressure:  - int p  nabla . w  dx
        F_pres = -dot(div(w), p) * dx

        # Continuity:  int q  nabla . u_new  dx
        # Note: use u_new (not u_theta) for the continuity constraint —
        # this enforces incompressibility at t^{n+1} exactly.
        F_cont = dot(q, div(u_new)) * dx

        # ── Total residual ────────────────────────────────────────────────────
        self.F = F_time + F_conv + F_diff + F_pres + F_cont

        # ── Jacobian via UFL automatic differentiation ────────────────────────
        # derivative(F, u_p) differentiates F w.r.t. the Function u_p.
        # Because the convection term is fully nonlinear in u_new = split(u_p)[0],
        # the Jacobian includes the consistent tangent from the convection term.
        # This is equivalent to Newton linearisation of the full nonlinear system.
        w_trial = ufl.TrialFunction(W)
        self.J  = derivative(self.F, self.u_p, w_trial)
