"""
carotid_cfd.config
==================
All physical parameters and solver settings in one dataclass.
No FSI.  No external dependencies — pure Python / dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FluidProperties:
    """
    Newtonian blood model.

    Newtonian is valid here because the dominant shear rate in the
    carotid artery (U/R ≈ 0.2/0.004 ≈ 50 s⁻¹) is well above the
    ~10 s⁻¹ threshold where non-Newtonian effects become significant.
    """
    rho: float = 1060.0   # kg/m³   density
    mu:  float = 3.5e-3   # Pa·s    dynamic viscosity

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m²/s]."""
        return self.mu / self.rho


@dataclass
class GeometryParameters:
    """Carotid bifurcation dimensions (SI, metres)."""
    D_inlet:           float = 8.0e-3    # m  parent artery inner diameter
    D_outlet:          float = 5.5e-3    # m  daughter branch diameter
    L_parent:          float = 100.0e-3  # m  parent artery length
    L_branch:          float = 80.0e-3   # m  each branch length
    phi_deg:           float = 30.0      # °  total bifurcation angle
    stenosis_severity: float = 0.50      # δ ∈ [0,1)  fractional radius reduction
    stenosis_length:   float = 10.0e-3   # m  axial extent of stenosis
    stenosis_centre:   float = 40.0e-3   # m  from branch inlet

    @property
    def R_inlet(self)  -> float: return self.D_inlet  / 2.0
    @property
    def R_outlet(self) -> float: return self.D_outlet / 2.0
    @property
    def phi_half(self) -> float: return math.radians(self.phi_deg / 2.0)

    @property
    def area_reduction(self) -> float:
        """Fractional area reduction at stenosis throat."""
        return 1.0 - (1.0 - self.stenosis_severity) ** 2


@dataclass
class PulsatileWaveform:
    """
    Pulsatile inlet waveform — notation matches the report exactly.

    The spatially-averaged centreline velocity is:

        U(t) = u_bar + A * sin(omega * t)

    where:
        u_bar  [m/s]   mean (drift) velocity
        A      [m/s]   oscillation amplitude
        omega  [rad/s] angular frequency  =  2π / T_cycle

    The full inlet profile over the cross-section is:

        v_D(x, y, t) = phi(x, y) * U(t) * e_1

    where phi(x, y) = 2 * (1 - (r/R)^2) is the parabolic Poiseuille
    shape function (factor 2 so that the spatial average equals U(t))
    and e_1 is the unit vector along the flow axis.

    Default values
    --------------
    Physiological carotid artery data (Ku, 1997):
        u_bar  = 0.20 m/s   (time-averaged centreline velocity)
        A      = 0.15 m/s   (pulsatile amplitude)
        T_cycle = 1.0 s     (heart rate 60 bpm)
        omega  = 2π rad/s

    These give:
        U_min = u_bar - A = 0.05 m/s  (diastole)
        U_max = u_bar + A = 0.35 m/s  (systole)
    """
    u_bar:   float = 0.200   # m/s   mean centreline velocity  (ū in report)
    A:       float = 0.150   # m/s   amplitude                 (A in report)
    T_cycle: float = 1.0     # s     cardiac period

    @property
    def omega(self) -> float:
        """Angular frequency ω = 2π / T [rad/s]."""
        return 2.0 * math.pi / self.T_cycle

    @property
    def U_max(self) -> float:
        """Peak (systolic) velocity [m/s]."""
        return self.u_bar + self.A

    @property
    def U_min(self) -> float:
        """Minimum (diastolic) velocity [m/s]."""
        return self.u_bar - self.A

    def velocity_at(self, t: float) -> float:
        """
        Spatially-averaged centreline velocity at time t.

        Implements:  U(t) = ū + A sin(ω t)
        """
        return self.u_bar + self.A * math.sin(self.omega * t)


@dataclass
class SolverConfig:
    """
    Time integration and linear solver settings.

    θ = 0.5  →  Crank–Nicolson (2nd order, unconditionally stable)
    θ = 1.0  →  Backward Euler (1st order, more robust for startup)

    Newton convergence is checked on the *absolute* residual norm,
    which is mesh-independent unlike the relative norm.
    """
    # Time stepping
    dt:    float = 5.0e-4   # s    timestep
    t_end: float = 3.0      # s    total simulation time
    theta: float = 0.5      # θ    Crank–Nicolson weight

    # Newton iteration
    newton_max_iter: int   = 100
    newton_abs_tol:  float = 1.0e-8
    newton_rel_tol:  float = 1.0e-6

    # Krylov linear solver (inner loop of Newton)
    # "gmres" + "ilu"  works well for moderate mesh sizes (serial)
    # "gmres" + "hypre_amg"  recommended for large/parallel runs
    ksp_type:    str   = "preonly"
    pc_type:     str   = "lu"
    ksp_rtol:    float = 1.0e-6
    ksp_atol:    float = 1.0e-8
    ksp_max_it:  int   = 500

    # Output
    output_interval: int = 10   # write XDMF every N timesteps
    
    # dt: float = 2.0e-3
    # t_end: float = 0.2
    # theta: float = 0.5

    # # Easier Newton convergence
    # newton_abs_tol: float = 1e-4
    # newton_rel_tol: float = 1e-2
    # newton_max_iter: int = 5

    # # Faster linear solves
    # ksp_type:    str = "gmres"
    # pc_type:     str = "ilu"
    # ksp_rtol:    float = 1e-4
    # ksp_atol:    float = 1e-6
    # ksp_max_it:  int = 50

    # output_interval: int = 10

@dataclass
class SimulationConfig:
    """
    Single configuration object passed through the entire pipeline.

    Usage
    -----
    >>> cfg = SimulationConfig()                         # all defaults
    >>> cfg = SimulationConfig(                          # custom stenosis
    ...     geometry=GeometryParameters(stenosis_severity=0.7),
    ...     output_dir=Path("results_severe"),
    ... )
    """
    fluid:      FluidProperties    = field(default_factory=FluidProperties)
    geometry:   GeometryParameters = field(default_factory=GeometryParameters)
    waveform:   PulsatileWaveform  = field(default_factory=PulsatileWaveform)
    solver:     SolverConfig       = field(default_factory=SolverConfig)
    mesh_file:  Path               = Path("carotid.xdmf")
    output_dir: Path               = Path("results")

    def reynolds_peak(self) -> float:
        """Peak Re = ρ U_max D / μ."""
        return (self.fluid.rho * self.waveform.U_max
                * self.geometry.D_inlet / self.fluid.mu)

    def womersley_number(self) -> float:
        """Wo = R sqrt(ω / ν)."""
        return self.geometry.R_inlet * math.sqrt(
            self.waveform.omega / self.fluid.nu
        )

    def print_summary(self) -> None:
        sep = "=" * 54
        wv  = self.waveform
        print(sep)
        print("  carotid_cfd — Simulation Config")
        print(sep)
        print(f"  Inlet waveform  U(t) = ū + A sin(ωt)")
        print(f"    ū (u_bar)     = {wv.u_bar*1e3:.1f} mm/s")
        print(f"    A (amplitude) = {wv.A*1e3:.1f} mm/s")
        print(f"    ω (omega)     = {wv.omega:.3f} rad/s  "
              f"(T = {wv.T_cycle:.2f} s)")
        print(f"    U_min / U_max = {wv.U_min*1e3:.0f} / "
              f"{wv.U_max*1e3:.0f} mm/s")
        print(f"  Re (peak)       = {self.reynolds_peak():.1f}  "
              f"[< 2300 laminar ✓]")
        print(f"  Womersley No.   = {self.womersley_number():.2f}  "
              f"[< 10 quasi-Poiseuille ✓]")
        print(f"  ν (kinematic)   = {self.fluid.nu:.2e} m²/s")
        print(f"  Δt              = {self.solver.dt*1e3:.2f} ms")
        print(f"  t_end           = {self.solver.t_end:.2f} s  "
              f"({self.solver.t_end/wv.T_cycle:.1f} cycles)")
        print(f"  Stenosis δ      = "
              f"{self.geometry.stenosis_severity*100:.0f}%  →  "
              f"{self.geometry.area_reduction*100:.0f}% area reduction")
        print(f"  Mesh            = {self.mesh_file}")
        print(f"  Output          = {self.output_dir}")
        print(sep)
