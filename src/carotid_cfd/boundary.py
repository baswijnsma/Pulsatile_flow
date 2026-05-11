"""
carotid_cfd.boundary
====================
Boundary condition construction — dolfinx API.

Boundary conditions implemented (matches the report exactly)
------------------------------------------------------------

1. Inlet  Gamma_in  —  DIRICHLET on velocity
   v = v_D(t) = phi(x,y) * (u_bar + A sin(omega t)) * e_1

   phi(x,y) = 2 * (1 - (r/R)^2)   parabolic Poiseuille shape
   factor 2 ensures the spatial average over the cross-section equals U(t)

2. Wall   Gamma_wall  —  DIRICHLET on velocity
   v = 0   (no-slip)

3. Outlets  Gamma_out1, Gamma_out2  —  NATURAL (no BC applied)
   sigma * n = 0   (stress-free / traction-free)

   This condition is NATURAL in the variational formulation.
   When you integrate the diffusion term by parts and impose no
   Dirichlet BC on the outlets, the boundary integral

       integral_{Gamma_out} w . (sigma n) dS

   vanishes automatically because sigma n = 0 there.
   Nothing is added to the code for the outlets.

dolfinx BC pattern (three steps for Function-valued BCs)
---------------------------------------------------------
    # 1. Find facets with the boundary tag
    facets = facet_tags.find(tag)

    # 2. Locate DOFs — two-argument form needed when value is a Function
    dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

    # 3. Build BC with value in the COLLAPSED space V
    bc = fem.dirichletbc(value_function_in_V, dofs, W.sub(0))

Where to find / change inlet parameter values
---------------------------------------------
All inlet parameters live in SimulationConfig.waveform (PulsatileWaveform):

    cfg.waveform.u_bar    — mean velocity ū  [m/s]     default 0.200
    cfg.waveform.A        — amplitude A      [m/s]     default 0.150
    cfg.waveform.T_cycle  — cardiac period T [s]       default 1.0
    cfg.waveform.omega    — angular freq ω   [rad/s]   = 2π/T  (computed)

To change them, pass a custom PulsatileWaveform when building SimulationConfig:

    from carotid_cfd import SimulationConfig, PulsatileWaveform
    cfg = SimulationConfig(
        waveform=PulsatileWaveform(
            u_bar   = 0.20,   # m/s  mean centreline velocity
            A       = 0.15,   # m/s  pulsatile amplitude
            T_cycle = 1.0,    # s    one cardiac cycle
        )
    )

The parabolic shape R (inlet radius) comes from cfg.geometry.R_inlet.
The flow direction is always e_1 (x-axis, flow_dir=0).
"""

from __future__ import annotations

import numpy as np

from dolfinx.mesh import Mesh, MeshTags
import dolfinx.fem as fem
import dolfinx

from carotid_cfd.config import SimulationConfig, PulsatileWaveform
from carotid_cfd.tags import BoundaryTag


# ── Inlet velocity profile ─────────────────────────────────────────────────────

def make_inlet_updater(
    u_inlet:  fem.Function,
    wave:     PulsatileWaveform,
    R:        float,
    flow_dir: int = 0,
):
    """
    Return an updater(t) callable that refreshes the inlet Function.

    Implements the report's inlet condition:
        v_D(x, y, t) = phi(x, y) * U(t) * e_1

    where:
        U(t)      = u_bar + A * sin(omega * t)   [scalar, time-varying]
        phi(x, y) = 2 * (1 - (r/R)^2)           [parabolic shape, spatial]
        e_1       = unit vector along flow axis

    The factor 2 in phi ensures:
        (1/|Gamma_in|) * integral phi dA = 1
    so the cross-section-averaged velocity exactly equals U(t).

    Parameters
    ----------
    u_inlet  : fem.Function in collapsed V  — updated by this function
    wave     : PulsatileWaveform            — holds u_bar, A, omega
    R        : float                        — inlet radius [m]
    flow_dir : int                          — axis index: 0=x, 1=y, 2=z

    Returns
    -------
    updater  : callable(t: float) -> None
               Call this before each Newton solve to refresh u_inlet.
    """
    # Mutable container — lets the closure mutate t without nonlocal
    state = {"t": 0.0}

    def _profile(x: np.ndarray) -> np.ndarray:
        """
        Interpolation function for dolfinx.

        Receives x of shape (3, N) — coordinates of N DOF points.
        Returns  array of shape (gdim, N) — velocity vector at each point.
        """
        # Radial distance from the centreline
        # (sum of squared components on the non-flow axes)
        r2 = sum(x[i] ** 2 for i in range(3) if i != flow_dir)
        r  = np.sqrt(r2)

        # U(t) = ū + A sin(ω t)
        U_t = wave.velocity_at(state["t"])

        # phi(r) * U(t) — clamp to zero outside the inlet disk (safety)
        speed = 2.0 * U_t * np.maximum(0.0, 1.0 - (r / R) ** 2)

        gdim   = u_inlet.function_space.mesh.geometry.dim
        values = np.zeros((gdim, x.shape[1]), dtype=np.float64)
        values[flow_dir] = speed
        return values

    def updater(t: float) -> None:
        state["t"] = t
        u_inlet.interpolate(_profile)

    # Initialise at t = 0
    updater(0.0)
    return updater


# ── Boundary condition container ──────────────────────────────────────────────

class BoundaryConditions:
    """
    Constructs and holds all Dirichlet BCs.

    Only TWO Dirichlet conditions are imposed (matching the report):
      - Inlet:  v = v_D(t)  on Gamma_in
      - Wall:   v = 0       on Gamma_wall

    The outlet condition sigma*n = 0 is NATURAL — no BC object is created.

    The inlet Function u_inlet is updated in-place each timestep via
    updater(t).  Because DirichletBC holds a reference to u_inlet,
    the BC automatically uses the new values on the next assembly.
    No BC objects are rebuilt between timesteps.

    Usage
    -----
    >>> bcs = BoundaryConditions(spaces, mesh, facet_tags, cfg)
    >>> bcs.update(t)            # call every timestep before solve
    >>> problem = NonlinearProblem(F, u_p, bcs=bcs.list, ...)
    """

    def __init__(
        self,
        spaces:     dict,
        mesh:       Mesh,
        facet_tags: MeshTags,
        cfg:        SimulationConfig,
    ):
        W    = spaces["W"]
        V    = spaces["V"]
        fdim = mesh.topology.dim - 1

        def _dofs_velocity(tag: int):
            """
            Locate velocity DOFs on tagged facets.

            Uses the two-argument form of locate_dofs_topological because
            the BC value is a Function in the collapsed space V (not a
            scalar Constant).  The two-argument form maps DOFs between
            W.sub(0) (the sub-space) and V (the collapsed space).
            """
            facets = facet_tags.find(np.int32(tag))
            return fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

        # ── BC 1: Inlet — pulsatile parabolic profile ─────────────────────────
        # v = phi(x,y) * (u_bar + A sin(omega t)) * e_1   on Gamma_in
        # u_inlet lives in the COLLAPSED velocity space V.
        self.u_inlet = fem.Function(V, name="u_inlet")
        self._updater = make_inlet_updater(
            self.u_inlet,
            cfg.waveform,
            cfg.geometry.R_inlet,
            flow_dir=0,
        )
        bc_inlet = fem.dirichletbc(
            self.u_inlet,
            _dofs_velocity(BoundaryTag.INLET),
            W.sub(0),
        )

        # ── BC 2: Wall — no-slip ──────────────────────────────────────────────
        # v = 0   on Gamma_wall
        # zero_v is a zero Function in V (never changes, no updater needed).
        zero_v = fem.Function(V, name="zero_wall")
        bc_wall = fem.dirichletbc(
            zero_v,
            _dofs_velocity(BoundaryTag.WALL),
            W.sub(0),
        )

        # ── Outlets: NO BC — sigma*n = 0 is natural ───────────────────────────
        # Nothing is applied at OUTLET_1 or OUTLET_2.
        # The stress-free condition emerges from the variational form
        # when no Dirichlet BC is imposed on those boundaries.

        self.list: list[fem.DirichletBC] = [bc_inlet, bc_wall]

    def update(self, t: float) -> None:
        """
        Refresh the inlet profile for time t.
        Call this once per timestep before calling problem.solve().
        """
        self._updater(t)
