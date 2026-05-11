"""
carotid_cfd.tags
================
Integer boundary and subdomain tags.

These constants are the single source of truth shared between
build_mesh.py (Gmsh physical groups) and the solver.  Every
DirichletBC and ds() measure references these — never raw integers.

Must match exactly the Physical Group IDs assigned in build_mesh.py.
"""


class BoundaryTag:
    """Facet (surface) tags — match Gmsh Physical Groups on surfaces."""
    INLET    = 1   # parent artery inlet cross-section
    OUTLET_1 = 2   # upper (internal carotid) outlet
    OUTLET_2 = 3   # lower (external carotid) outlet
    WALL     = 4   # inner arterial wall (no-slip)


class DomainTag:
    """Cell (volume) tags — match Gmsh Physical Groups on volumes."""
    FLUID = 1
