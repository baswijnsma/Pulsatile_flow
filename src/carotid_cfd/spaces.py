"""
carotid_cfd.spaces
==================
Function space construction — dolfinx + basix API.

Taylor-Hood P2/P1 satisfies the inf-sup (LBB) condition exactly.
No pressure stabilisation needed.

Key dolfinx pattern
-------------------
    V, WV_map = W.sub(0).collapse()
    Q, WQ_map = W.sub(1).collapse()

collapse() returns (FunctionSpace, dof_index_map).
The dof maps WV_map and WQ_map are needed to extract velocity and
pressure arrays from the mixed solution:

    u_out.x.array[:] = u_p.x.array[WV_map]
    p_out.x.array[:] = u_p.x.array[WQ_map]

This is the pattern from the reference implementation and avoids
calling .collapse() (which allocates a new Function) at every timestep.
"""

from __future__ import annotations

from dolfinx.mesh import Mesh
import dolfinx.fem as fem
import basix.ufl
import ufl
import numpy as np


def build_spaces(mesh: Mesh) -> dict:
    """
    Build Taylor-Hood P2/P1 mixed function spaces.

    Returns
    -------
    dict with keys:
        "W"      — mixed space (velocity x pressure)  — used for the solve
        "V"      — collapsed velocity space [P2]^d    — for BCs, ICs, output
        "Q"      — collapsed pressure space P1         — for output
        "WV_map" — int array mapping W dofs → V dofs  — for array extraction
        "WQ_map" — int array mapping W dofs → Q dofs  — for array extraction
    """
    gdim      = mesh.geometry.dim
    cell_name = mesh.topology.cell_name()

    # Elements: basix.ufl replaces the legacy ufl VectorElement/FiniteElement
    P2 = basix.ufl.element("Lagrange", cell_name, 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", cell_name, 1)
    TH = basix.ufl.mixed_element([P2, P1])

    #debug
    # P1v = basix.ufl.element("Lagrange", cell_name, 1, shape=(gdim,))
    # P1p = basix.ufl.element("Lagrange", cell_name, 1)
    # TH  = basix.ufl.mixed_element([P1v, P1p])

    W = fem.functionspace(mesh, TH)

    # collapse() returns (FunctionSpace, dof_index_map)
    # Keep BOTH — the dof maps are needed for efficient array extraction
    V, WV_map = W.sub(0).collapse()
    Q, WQ_map = W.sub(1).collapse()

    n_tot = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    print(f"[spaces]  Taylor-Hood P2/P1:  {n_tot:,} total DOFs")
    print(f"          velocity {V.dofmap.index_map.size_global * gdim:,}"
          f"   pressure {Q.dofmap.index_map.size_global:,}")

    return {
        "W":      W,
        "V":      V,
        "Q":      Q,
        "WV_map": WV_map,
        "WQ_map": WQ_map,
    }
