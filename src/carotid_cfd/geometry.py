"""
carotid_cfd.geometry
====================
Mesh loading using the dolfinx XDMF API.

Key dolfinx patterns used here
-------------------------------

1. Reading mesh + meshtags from XDMF
   -----------------------------------
   dolfin (OLD):
       mesh = Mesh()
       with XDMFFile("mesh.xdmf") as f:
           f.read(mesh)
       mvc = MeshValueCollection("size_t", mesh, 2)
       with XDMFFile("boundaries.xdmf") as f:
           f.read(mvc)
       tags = cpp.mesh.MeshTags_int32(mesh, mvc)

   dolfinx (NEW):
       with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as f:
           mesh = f.read_mesh(name="Grid")
           mesh.topology.create_connectivity(fdim, tdim)
           tags = f.read_meshtags(mesh, name="Grid")
       # IMPORTANT: meshtags must be read from the SAME file
       # they were written to — they share the same node numbering.
       # Boundary tags live in a separate file, opened separately.

2. Connectivity must be created before locate_dofs or meshtags
   -----------------------------------------------------------
   mesh.topology.create_connectivity(fdim, tdim)   # facets → cells
   mesh.topology.create_connectivity(tdim, fdim)   # cells  → facets

3. locate_entities_boundary replaces SubDomain
   --------------------------------------------
   dolfin (OLD):
       class Inlet(SubDomain):
           def inside(self, x, on_boundary):
               return on_boundary and near(x[0], 0.0)

   dolfinx (NEW):
       def inlet_marker(x):
           return np.isclose(x[0], 0.0, atol=1e-8)
       facets = locate_entities_boundary(mesh, fdim, inlet_marker)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    Mesh,
    MeshTags,
    meshtags,
    locate_entities_boundary,
)

from carotid_cfd.config import SimulationConfig
from carotid_cfd.tags import BoundaryTag


def load_mesh(
    cfg: SimulationConfig,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> tuple[Mesh, MeshTags, MeshTags]:
    """
    Load mesh, boundary facet tags, and cell domain tags from XDMF.

    File convention (produced by build_mesh.py)
    -------------------------------------------
    {cfg.mesh_file}                        volume cells
    {stem}_boundaries.xdmf                 boundary facets + BoundaryTag values
    {stem}_subdomains.xdmf                 volume cells   + DomainTag values
                                            (optional; empty if absent)

    Parameters
    ----------
    cfg  : SimulationConfig
    comm : MPI communicator

    Returns
    -------
    mesh        : dolfinx Mesh
    facet_tags  : MeshTags on facets (tdim-1), values = BoundaryTag.*
    cell_tags   : MeshTags on cells  (tdim),   values = DomainTag.*
    """
    mesh_path = Path(cfg.mesh_file)
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}\n"
            "Run  python build_mesh.py  first to generate XDMF files."
        )

    # ── 1. Read volume mesh ───────────────────────────────────────────────────
    with XDMFFile(comm, str(mesh_path), "r") as fh:
        mesh = fh.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # ── 2. Create connectivities needed by meshtags and DirichletBC ───────────
    # This is required BEFORE calling read_meshtags or locate_dofs_topological.
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)

    # ── 3. Read boundary facet tags ───────────────────────────────────────────
    bnd_path = mesh_path.with_name(mesh_path.stem + "_boundaries.xdmf")
    if bnd_path.exists():
        with XDMFFile(comm, str(bnd_path), "r") as fh:
            facet_tags = fh.read_meshtags(mesh, name="Grid")
    else:
        warnings.warn(
            f"Boundary tag file not found: {bnd_path}\n"
            "Boundary conditions will not be enforced correctly.\n"
            "Re-run build_mesh.py to regenerate.",
            stacklevel=2,
        )
        facet_tags = meshtags(
            mesh, fdim,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    # ── 4. Read cell domain tags (optional) ───────────────────────────────────
    sub_path = mesh_path.with_name(mesh_path.stem + "_subdomains.xdmf")
    if sub_path.exists():
        with XDMFFile(comm, str(sub_path), "r") as fh:
            cell_tags = fh.read_meshtags(mesh, name="Grid")
    else:
        cell_tags = meshtags(
            mesh, tdim,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    _print_info(mesh, facet_tags)
    return mesh, facet_tags, cell_tags


def build_facet_tags_from_geometry(
    mesh: Mesh,
    cfg:  SimulationConfig,
) -> MeshTags:
    """
    Build boundary MeshTags purely from geometry, without a pre-tagged mesh.

    Use this only for unit tests or simple meshes built directly in Python.
    For production runs, use tags embedded in the XDMF files from Gmsh.

    dolfinx pattern: locate_entities_boundary(mesh, fdim, marker_fn)
    where marker_fn(x) returns a bool array — replaces SubDomain.inside().
    np.isclose() replaces the legacy near() function.
    """
    fdim = mesh.topology.dim - 1
    g    = cfg.geometry

    all_facets: list[np.ndarray] = []
    all_tags:   list[np.ndarray] = []

    def _tag(marker_fn, tag: int) -> None:
        fcts = locate_entities_boundary(mesh, fdim, marker_fn)
        all_facets.append(fcts)
        all_tags.append(np.full(len(fcts), tag, dtype=np.int32))

    # Inlet:  x ≈ 0
    _tag(lambda x: np.isclose(x[0], 0.0, atol=1e-8), BoundaryTag.INLET)

    # Upper outlet: near tip of upper branch
    cx1 = g.L_parent + g.L_branch * np.cos(g.phi_half)
    cy1 = g.L_branch * np.sin(g.phi_half)
    _tag(
        lambda x, cx=cx1, cy=cy1: (
            np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2) < g.R_outlet * 1.5
        ),
        BoundaryTag.OUTLET_1,
    )

    # Lower outlet: near tip of lower branch
    cy2 = -g.L_branch * np.sin(g.phi_half)
    _tag(
        lambda x, cx=cx1, cy=cy2: (
            np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2) < g.R_outlet * 1.5
        ),
        BoundaryTag.OUTLET_2,
    )

    # Wall: everything else on the boundary is the wall
    # (combine all above to find the complement via difference)
    all_boundary = locate_entities_boundary(
        mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    tagged_so_far = np.concatenate(all_facets) if all_facets else np.array([], dtype=np.int32)
    wall_facets = np.setdiff1d(all_boundary, tagged_so_far)
    all_facets.append(wall_facets)
    all_tags.append(np.full(len(wall_facets), BoundaryTag.WALL, dtype=np.int32))

    facets = np.concatenate(all_facets).astype(np.int32)
    tags   = np.concatenate(all_tags).astype(np.int32)
    idx    = np.argsort(facets)    # meshtags requires sorted indices
    return meshtags(mesh, fdim, facets[idx], tags[idx])


def _print_info(mesh: Mesh, facet_tags: MeshTags) -> None:
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    n_verts = mesh.topology.index_map(0).size_local
    print(f"[geometry] Mesh:  {n_cells:,} cells,  {n_verts:,} vertices")
    vals = facet_tags.values
    for tag, name in [
        (BoundaryTag.INLET,    "inlet   "),
        (BoundaryTag.OUTLET_1, "outlet_1"),
        (BoundaryTag.OUTLET_2, "outlet_2"),
        (BoundaryTag.WALL,     "wall    "),
    ]:
        n = int(np.sum(vals == tag))
        if n:
            print(f"           {name}: {n:,} facets (tag={tag})")
