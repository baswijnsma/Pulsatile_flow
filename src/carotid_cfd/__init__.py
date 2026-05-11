"""
carotid_cfd
===========
Pulsatile incompressible Navier-Stokes solver for a bifurcated
carotid artery with parametric stenosis.

Built on dolfinx (FEniCSx).  Rigid walls.  Laminar flow.
No FSI.

Quick start
-----------
    from carotid_cfd import SimulationConfig, run_simulation
    monitor = run_simulation(SimulationConfig())

Sub-modules
-----------
config    — FluidProperties, GeometryParameters, PulsatileWaveform,
            SolverConfig, SimulationConfig
tags      — BoundaryTag, DomainTag
geometry  — load_mesh(), build_facet_tags_from_geometry()
spaces    — build_spaces()
boundary  — BoundaryConditions, make_inlet_updater()
weakform  — NavierStokesForm, epsilon()
postproc  — FlowMonitor
solver    — NewtonSolver, run_simulation()
"""

from carotid_cfd.config import (        # noqa: F401
    FluidProperties,
    GeometryParameters,
    PulsatileWaveform,
    SolverConfig,
    SimulationConfig,
)
from carotid_cfd.tags import (          # noqa: F401
    BoundaryTag,
    DomainTag,
)
from carotid_cfd.geometry import (      # noqa: F401
    load_mesh,
    build_facet_tags_from_geometry,
)
from carotid_cfd.spaces import (        # noqa: F401
    build_spaces,
)
from carotid_cfd.boundary import (      # noqa: F401
    BoundaryConditions,
    make_inlet_updater,
)
from carotid_cfd.weakform import (      # noqa: F401
    NavierStokesForm,
    epsilon,
)
from carotid_cfd.postproc import (      # noqa: F401
    FlowMonitor,
)
from carotid_cfd.solver import (        # noqa: F401
    run_simulation,
)

__all__ = [
    "FluidProperties", "GeometryParameters", "PulsatileWaveform",
    "SolverConfig", "SimulationConfig",
    "BoundaryTag", "DomainTag",
    "load_mesh", "build_facet_tags_from_geometry",
    "build_spaces",
    "BoundaryConditions", "make_inlet_updater",
    "NavierStokesForm", "epsilon",
    "FlowMonitor",
    "run_simulation",
]
