"""
=============================================================================
build_mesh.py  —  Gmsh geometry and mesh builder
=============================================================================
4EM60 — Advanced Discretization Techniques
Pulsatile flow in a bifurcated carotid artery with stenosis

Geometry improvements over the previous version
------------------------------------------------
1. SMOOTH BIFURCATION via OCC ThruSections loft
   Previously: three cylinders Boolean-fused → sharp junction seams.
   Now:
     • Parent artery: straight cylinder (already smooth).
     • Junction transition: each branch is a ThruSections loft through
       three circular cross-sections:
         - Profile 0 at x = L_parent     radius R_in   (matches parent)
         - Profile 1 at x = midpoint      radius lerp(R_in → R_out)
         - Profile 2 at x = branch tip    radius R_out
       The loft produces a C¹-continuous surface that morphs the large
       parent circle smoothly into the smaller branch circle over the
       branch length.
     • Fusing parent + two lofted branches gives a watertight Y-junction
       with no sharp edges at the split point.

2. SPHERE-CAP PLAQUE MODEL
   Previously: cosine-profile revolution symmetric around the artery axis.
     → not realistic (plaques are focal, eccentric, on one wall side)
   Now:
     • Each plaque is defined by a sphere intersected with the lumen.
     • Parameters per plaque:
         R_sphere  — sphere radius [mm]     (controls lateral extent)
         h         — intrusion depth [mm]   (how far it protrudes into lumen)
                     constraint:  0 < h < R_sphere
         x_pos     — axial position [mm]    (along branch axis)
         angle_deg — circumferential angle  (0=+y, 90=+z, 180=-y, 270=-z)
     • The sphere centre is placed at distance (R_sphere - h) outside the
       vessel wall, so exactly a cap of height h protrudes into the lumen.
     • Boolean workflow:  sphere ∩ vessel = plaque solid;
                          vessel - plaque = stenosed lumen.
     • Multiple plaques can be placed on the same branch by chaining the
       boolean cuts.
     • Cross-section geometry (useful for reporting severity):
         The cap intersects the wall at a circle of radius
           r_cap = sqrt(R_sphere² - (R_sphere - h)²) = sqrt(2 R_sphere h - h²)
         The lumen blockage can be quantified as the overlap area of
         disk(r_cap) and disk(R_vessel) at the wall surface.

Physical group tags (match BoundaryTag in carotid_cfd/tags.py)
--------------------------------------------------------------
  Surface:  1 → INLET   2 → OUTLET_1   3 → OUTLET_2   4 → WALL
  Volume:   1 → FLUID

Usage
-----
  python build_mesh.py                          # default: δ=0.5
  python build_mesh.py --stenosis 0.0           # healthy
  python build_mesh.py --stenosis 0.7           # severe
  python build_mesh.py --lc 0.8                 # finer mesh
  python build_mesh.py --study                  # mesh independence study
  python build_mesh.py --out my_mesh            # custom output name

Dependencies
------------
  pip install gmsh meshio numpy
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("DISPLAY", "")   # headless gmsh

try:
    import gmsh
except ImportError:
    sys.exit("gmsh not found — run:  pip install gmsh")

try:
    import meshio
except ImportError:
    sys.exit("meshio not found — run:  pip install meshio")


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PlaqueConfig:
    """
    One spherical-cap plaque deposit.

    Parameters
    ----------
    R_sphere  : float   sphere radius [mm]
    h         : float   intrusion depth into lumen [mm],  0 < h < R_sphere
    x_pos     : float   axial position along the branch [mm]
    angle_deg : float   circumferential placement angle [°]
                        0   → top    (+y side of branch in branch-local frame)
                        90  → side   (+z)
                        180 → bottom (-y)
                        270 → other side (-z)

    Derived geometry
    ----------------
    r_cap = sqrt(2 * R_sphere * h - h²)   radius of the cap footprint at wall
    The lumen blockage fraction (area) depends on r_cap vs R_vessel.
    """
    R_sphere:  float = 3.5      # mm
    h:         float = 1.0      # mm  intrusion depth
    x_pos:     float = 40.0     # mm  along branch axis
    angle_deg: float = 0.0      # °   circumferential angle

    def r_cap(self) -> float:
        """Radius of the circular cap footprint at the wall surface."""
        return math.sqrt(max(0.0, 2 * self.R_sphere * self.h - self.h ** 2))

    def area_reduction_fraction(self, R_vessel: float) -> float:
        """
        Approximate lumen blockage as area of circular segment.

        Returns the fraction of the vessel cross-sectional area that is
        blocked by the plaque cap.  This is the overlap area of:
          - disk of radius r_cap centred at (0, R_vessel) — cap circle
          - disk of radius R_vessel centred at origin       — vessel

        Computed using the standard two-circle intersection formula.
        """
        r1  = R_vessel          # vessel radius
        r2  = self.r_cap()      # cap footprint radius
        d   = R_vessel          # distance between centres = R_vessel
        #                         (cap centre is at the wall surface)

        if r2 < 1e-9 or d >= r1 + r2:
            return 0.0
        if d + r1 <= r2:
            return 1.0   # vessel fully inside cap (extreme case)
        if d + r2 <= r1:
            # cap fully inside vessel
            return (math.pi * r2 ** 2) / (math.pi * r1 ** 2)

        # General: two-circle lens area
        cos_a1 = (d**2 + r1**2 - r2**2) / (2 * d * r1 + 1e-15)
        cos_a2 = (d**2 + r2**2 - r1**2) / (2 * d * r2 + 1e-15)
        cos_a1 = max(-1.0, min(1.0, cos_a1))
        cos_a2 = max(-1.0, min(1.0, cos_a2))
        alpha1 = math.acos(cos_a1)
        alpha2 = math.acos(cos_a2)
        area = (r1**2 * (alpha1 - math.sin(alpha1) * math.cos(alpha1))
              + r2**2 * (alpha2 - math.sin(alpha2) * math.cos(alpha2)))
        return area / (math.pi * r1 ** 2)


@dataclass
class MeshConfig:
    """
    All geometric and meshing parameters.
    Dimensions are in millimetres (Gmsh native); converted to metres on export.
    """
    # Vessel dimensions
    D_inlet:  float = 8.0    # mm  parent artery inner diameter
    D_outlet: float = 5.5    # mm  daughter branch diameter
    L_parent: float = 100.0  # mm  parent artery length
    L_branch: float = 80.0   # mm  each branch length
    phi_deg:  float = 30.0   # °   total bifurcation angle (each branch ±phi/2)

    # Plaques: list of PlaqueConfig per branch (index 0 = upper, 1 = lower)
    # Default: one moderate plaque on each branch
    plaques_branch0: list = field(default_factory=lambda: [
        PlaqueConfig(R_sphere=4.0, h=1.2, x_pos=40.0, angle_deg=0.0),
    ])
    plaques_branch1: list = field(default_factory=lambda: [
        PlaqueConfig(R_sphere=4.0, h=1.2, x_pos=40.0, angle_deg=0.0),
    ])

    # Mesh sizing (mm)
    lc_bulk:      float = 1.2   # bulk element size
    lc_junction:  float = 0.5   # refined at Y-junction
    lc_plaque:    float = 0.25  # refined at plaque surface
    lc_wall:      float = 0.4   # near-wall refinement

    # Output
    output_stem: Path = Path("carotid")

    @property
    def R_inlet(self)  -> float: return self.D_inlet  / 2.0
    @property
    def R_outlet(self) -> float: return self.D_outlet / 2.0
    @property
    def phi_half(self) -> float: return math.radians(self.phi_deg / 2.0)

    def describe_plaques(self) -> None:
        """Print plaque geometry summary to stdout."""
        R = self.R_outlet
        for bi, plaques in enumerate([self.plaques_branch0, self.plaques_branch1]):
            for pi, p in enumerate(plaques):
                af = p.area_reduction_fraction(R) * 100
                print(f"  Branch {bi} plaque {pi}: "
                      f"R_sphere={p.R_sphere:.1f}mm  h={p.h:.1f}mm  "
                      f"x={p.x_pos:.1f}mm  angle={p.angle_deg:.0f}°  "
                      f"r_cap={p.r_cap():.2f}mm  "
                      f"area_blocked≈{af:.1f}%")


# Physical group tag constants — must match carotid_cfd/tags.py
INLET    = 1
OUTLET_1 = 2
OUTLET_2 = 3
WALL     = 4
FLUID    = 1


# ════════════════════════════════════════════════════════════════════════════
# 1.  GEOMETRY BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_geometry(cfg: MeshConfig) -> int:
    """
    Build the full carotid bifurcation geometry.

    Returns
    -------
    fluid_vol : int — Gmsh volume tag of the fluid domain
    """
    occ = gmsh.model.occ
    R_in  = cfg.R_inlet
    R_out = cfg.R_outlet
    Lp    = cfg.L_parent
    Lb    = cfg.L_branch
    phi   = cfg.phi_half

    # ── 1.1  Parent artery (straight cylinder) ────────────────────────────────
    # The parent is kept as a cylinder because it IS circular along its length.
    # The smooth morphing happens only in the branch lofts.
    parent = occ.addCylinder(0, 0, 0,  Lp, 0, 0,  R_in)

    # ── 1.2  Branches via ThruSections loft ───────────────────────────────────
    #
    # Each branch is a lofted solid through three circular cross-sections:
    #
    #   Profile 0 — at the junction (x = Lp):
    #               radius = R_in  (continuous with parent wall)
    #               normal along branch axis
    #
    #   Profile 1 — at the midpoint of the branch:
    #               radius = lerp(R_in, R_out, 0.5)
    #               (smooth transition)
    #
    #   Profile 2 — at the branch tip (outlet):
    #               radius = R_out
    #
    # The loft is smooth (makeRuled=False) so the cross-sections interpolate
    # with C¹ continuity, producing a physiologically plausible taper.
    # At the junction, profile 0 is centred at (Lp, 0, 0) and has the same
    # radius as the parent end-face — so the fuse operation creates a smooth
    # Y-junction without a hard rim.
    #
    # The zAxis parameter orients the circles so their normals point along
    # the branch axis, ensuring the loft sweeps in the correct direction.

    branch_tags = []
    branch_axes = []
    for sign in (+1.0, -1.0):
        dx  = Lb * math.cos(phi)
        dy  = Lb * math.sin(phi) * sign
        ax  = (math.cos(phi), math.sin(phi) * sign, 0.0)
        branch_axes.append(ax)

        # Profile 0: junction face
        c0 = occ.addCircle(Lp,         0,         0,     R_in,                   zAxis=ax)
        # Profile 1: mid-branch (linear taper in radius)
        c1 = occ.addCircle(Lp + dx*0.5, dy * 0.5, 0,     0.5*(R_in + R_out),     zAxis=ax)
        # Profile 2: outlet face
        c2 = occ.addCircle(Lp + dx,     dy,        0,     R_out,                  zAxis=ax)

        loops  = [occ.addCurveLoop([c]) for c in (c0, c1, c2)]
        result = occ.addThruSections(loops, makeSolid=True, makeRuled=False)
        branch_tags.append(result[0][1])

    # ── 1.3  Fuse parent + branches into one fluid domain ────────────────────
    occ.synchronize()
    fused, _ = occ.fuse(
        [(3, parent)],
        [(3, t) for t in branch_tags],
        removeObject=True,
        removeTool=True,
    )
    occ.synchronize()
    fluid = fused[0][1]

    # ── 1.4  Add plaques to each branch ───────────────────────────────────────
    all_plaques = [cfg.plaques_branch0, cfg.plaques_branch1]

    for branch_idx, (plaques, sign, ax) in enumerate(
        zip(all_plaques, [+1.0, -1.0], branch_axes)
    ):
        for plaque in plaques:
            fluid = _apply_plaque(fluid, plaque, sign, ax, cfg)

    occ.synchronize()
    return fluid


def _apply_plaque(
    fluid_tag: int,
    p:         PlaqueConfig,
    sign:      float,
    branch_ax: tuple,
    cfg:       MeshConfig,
) -> int:
    """
    Cut a spherical-cap plaque out of the fluid domain.

    Sphere placement
    ----------------
    The branch runs from (L_parent, 0, 0) in direction branch_ax.
    For a plaque at axial position x_pos along the branch, the point
    on the branch centreline is:

        P_centre = (L_parent + x_pos*ax[0],
                    x_pos * ax[1],
                    0)

    We need to place the sphere so it just touches the wall and intrudes
    by depth h.  The wall of the branch at that axial position is at
    distance R_vessel from the centreline.

    The circumferential direction is given by angle_deg in the plane
    perpendicular to the branch axis.  We pick two basis vectors
    perpendicular to the branch axis: e_perp1 in the XY plane,
    e_perp2 = branch_ax x e_perp1.

    The outward wall normal at the plaque site is:
        n = cos(angle_deg) * e_perp1 + sin(angle_deg) * e_perp2

    The sphere centre is placed at:
        P_sphere = P_centre + R_vessel * n + (R_sphere - h) * n
                 = P_centre + (R_vessel + R_sphere - h) * n

    so the sphere surface just touches the wall at depth h.

    Boolean workflow
    ----------------
        plaque = intersect(sphere, fluid)   ← cap solid inside lumen
        fluid  = cut(fluid, plaque)         ← remove cap from lumen

    Returns the new fluid volume tag after cutting.
    """
    occ   = gmsh.model.occ
    R_out = cfg.R_outlet
    phi   = cfg.phi_half
    Lp    = cfg.L_parent

    # ── Centreline point on branch at axial position x_pos ────────────────
    # branch_ax = (cos phi, ±sin phi, 0)
    x_pos = p.x_pos
    cx = Lp + x_pos * branch_ax[0]
    cy =      x_pos * branch_ax[1]
    cz = 0.0

    # ── Perpendicular basis in the cross-section plane ────────────────────
    # e_ax = branch_ax (unit, already normalised)
    # e_perp1: in the plane containing branch_ax and Y-axis
    #          = component of Y perpendicular to branch_ax
    ax  = np.array(branch_ax)
    ey  = np.array([0.0, 1.0, 0.0]) if abs(ax[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    e1  = ey - np.dot(ey, ax) * ax
    e1 /= np.linalg.norm(e1)                         # e_perp1
    e2  = np.cross(ax, e1)                            # e_perp2

    # ── Outward normal at angle_deg ────────────────────────────────────────
    theta   = math.radians(p.angle_deg)
    n_wall  = math.cos(theta) * e1 + math.sin(theta) * e2   # unit outward normal

    # ── Sphere centre ─────────────────────────────────────────────────────
    offset  = R_out + p.R_sphere - p.h   # distance from centreline to sphere centre
    sc = np.array([cx, cy, cz]) + offset * n_wall

    # ── Build sphere and Boolean operations ───────────────────────────────
    sphere = occ.addSphere(float(sc[0]), float(sc[1]), float(sc[2]), p.R_sphere)
    occ.synchronize()

    # Intersect sphere with fluid domain → plaque solid
    plaque_result, _ = occ.intersect(
        [(3, sphere)], [(3, fluid_tag)],
        removeObject=True, removeTool=False,
    )
    occ.synchronize()

    if not plaque_result:
        print(f"  [plaque] WARNING: sphere at ({sc[0]:.1f},{sc[1]:.1f}) "
              f"did not intersect the lumen — skipping.")
        # Clean up orphaned sphere if it survived
        try:
            occ.remove([(3, sphere)], recursive=True)
        except Exception:
            pass
        occ.synchronize()
        return fluid_tag

    plaque_tag = plaque_result[0][1]

    # Cut plaque from fluid domain
    stenosed, _ = occ.cut(
        [(3, fluid_tag)], [(3, plaque_tag)],
        removeObject=True, removeTool=True,
    )
    occ.synchronize()

    if not stenosed:
        print(f"  [plaque] WARNING: cut returned empty result — skipping.")
        return fluid_tag

    new_fluid = stenosed[0][1]
    af = p.area_reduction_fraction(R_out) * 100
    print(f"  [plaque] branch_sign={sign:+.0f}  x={x_pos:.1f}mm  "
          f"angle={p.angle_deg:.0f}°  r_cap={p.r_cap():.2f}mm  "
          f"~{af:.0f}% area blocked")
    return new_fluid


# ════════════════════════════════════════════════════════════════════════════
# 2.  BOUNDARY IDENTIFICATION
# ════════════════════════════════════════════════════════════════════════════

def identify_and_tag_boundaries(fluid_vol: int, cfg: MeshConfig) -> None:
    """
    Identify boundary surfaces by centroid position and assign
    Gmsh Physical Groups.

    Classification
    --------------
    Inlet:    centroid x ≈ 0   (parent artery entrance)
    Outlet 1: centroid near upper branch tip
    Outlet 2: centroid near lower branch tip
    Wall:     everything else

    The centroid classification is robust to small CAD kernel variations
    because it uses geometric position rather than entity numbering.
    """
    model = gmsh.model
    Lp    = cfg.L_parent
    Lb    = cfg.L_branch
    phi   = cfg.phi_half

    # Expected outlet centroids (mm)
    cx_out = Lp + Lb * math.cos(phi)
    cy_out1 = +Lb * math.sin(phi)
    cy_out2 = -Lb * math.sin(phi)
    tol = max(cfg.D_inlet, cfg.D_outlet) * 0.8

    boundary = model.getBoundary([(3, fluid_vol)], oriented=False, combined=False)
    surf_tags = [abs(s[1]) for s in boundary]

    inlet_s = []; out1_s = []; out2_s = []; wall_s = []

    for st in surf_tags:
        xn, yn, _, xx, yx, _ = model.getBoundingBox(2, st)
        cx = 0.5 * (xn + xx)
        cy = 0.5 * (yn + yx)

        if abs(cx) < tol and abs(cy) < tol:
            inlet_s.append(st)
        elif abs(cx - cx_out) < tol and abs(cy - cy_out1) < tol:
            out1_s.append(st)
        elif abs(cx - cx_out) < tol and abs(cy - cy_out2) < tol:
            out2_s.append(st)
        else:
            wall_s.append(st)

    if not inlet_s:
        raise RuntimeError("Inlet surface not found — check L_parent / tol")
    if not out1_s:
        raise RuntimeError("Outlet 1 surface not found")
    if not out2_s:
        raise RuntimeError("Outlet 2 surface not found")

    print(f"[boundaries] inlet={len(inlet_s)}  "
          f"out1={len(out1_s)}  out2={len(out2_s)}  wall={len(wall_s)} surfaces")

    model.addPhysicalGroup(2, inlet_s, INLET);    model.setPhysicalName(2, INLET,    "Inlet")
    model.addPhysicalGroup(2, out1_s, OUTLET_1);  model.setPhysicalName(2, OUTLET_1, "Outlet1")
    model.addPhysicalGroup(2, out2_s, OUTLET_2);  model.setPhysicalName(2, OUTLET_2, "Outlet2")
    model.addPhysicalGroup(2, wall_s, WALL);      model.setPhysicalName(2, WALL,     "Wall")
    model.addPhysicalGroup(3, [fluid_vol], FLUID); model.setPhysicalName(3, FLUID,   "Fluid")


# ════════════════════════════════════════════════════════════════════════════
# 3.  MESH SIZING FIELDS
# ════════════════════════════════════════════════════════════════════════════

def add_refinement_fields(cfg: MeshConfig) -> None:
    """
    Add distance-based mesh refinement fields.

    Field hierarchy
    ---------------
    F_junction : fine near the Y-junction (complex geometry, highest gradients)
    F_plaque   : fine near each plaque location
    F_combined : minimum of all above → background mesh
    """
    model = gmsh.model
    field = model.mesh.field
    occ   = model.occ
    Lp    = cfg.L_parent
    phi   = cfg.phi_half

    # ── Junction refinement ───────────────────────────────────────────────────
    pt_junc = occ.addPoint(Lp, 0, 0, cfg.lc_junction)
    occ.synchronize()

    f_d1 = field.add("Distance")
    field.setNumbers(f_d1, "PointsList", [pt_junc])

    f_t1 = field.add("Threshold")
    field.setNumber(f_t1, "InField",  f_d1)
    field.setNumber(f_t1, "SizeMin",  cfg.lc_junction)
    field.setNumber(f_t1, "SizeMax",  cfg.lc_bulk)
    field.setNumber(f_t1, "DistMin",  3.0)
    field.setNumber(f_t1, "DistMax",  15.0)

    all_thresh = [f_t1]

    # ── Plaque refinement (one field per plaque site) ─────────────────────────
    for sign, plaques in [(+1, cfg.plaques_branch0), (-1, cfg.plaques_branch1)]:
        for p in plaques:
            ax = (math.cos(phi), math.sin(phi) * sign, 0.0)
            x  = Lp + p.x_pos * ax[0]
            y  =      p.x_pos * ax[1]
            pt = occ.addPoint(x, y, 0, cfg.lc_plaque)
            occ.synchronize()

            f_dp = field.add("Distance")
            field.setNumbers(f_dp, "PointsList", [pt])

            f_tp = field.add("Threshold")
            field.setNumber(f_tp, "InField",  f_dp)
            field.setNumber(f_tp, "SizeMin",  cfg.lc_plaque)
            field.setNumber(f_tp, "SizeMax",  cfg.lc_bulk)
            field.setNumber(f_tp, "DistMin",  p.R_sphere * 0.5)
            field.setNumber(f_tp, "DistMax",  p.R_sphere * 3.0)
            all_thresh.append(f_tp)

    # ── Combined minimum field ────────────────────────────────────────────────
    f_min = field.add("Min")
    field.setNumbers(f_min, "FieldsList", all_thresh)
    field.setAsBackgroundMesh(f_min)


# ════════════════════════════════════════════════════════════════════════════
# 4.  MESH GENERATION AND XDMF EXPORT
# ════════════════════════════════════════════════════════════════════════════

def generate_and_export(cfg: MeshConfig) -> None:
    """
    Generate the tetrahedral mesh and export three XDMF files:
      {stem}.xdmf              — volume mesh (tetrahedra)
      {stem}_boundaries.xdmf  — boundary facets with BoundaryTag values
      {stem}_subdomains.xdmf  — volume cells  with DomainTag values

    Coordinates are scaled mm → m on export (factor 1e-3).
    """
    model = gmsh.model

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Optimize",    1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    print("\n[mesh] Generating 3D mesh …")
    model.mesh.generate(3)
    model.mesh.optimize("Netgen")
    print("[mesh] Done.")

    msh_path = str(cfg.output_stem) + ".msh"
    gmsh.write(msh_path)
    print(f"[mesh] Written: {msh_path}")
    gmsh.finalize()

    print("[mesh] Converting to XDMF (mm → m) …")
    _msh_to_xdmf(msh_path, cfg)


def _msh_to_xdmf(msh_path: str, cfg: MeshConfig) -> None:
    """
    Convert Gmsh .msh → dolfinx-compatible XDMF files via meshio.

    The cell_data_dict in meshio groups tags per cell type.  When a .msh file
    contains multiple blocks of the same type (e.g. multiple triangle blocks
    from different physical groups), we concatenate them here.
    """
    stem = str(cfg.output_stem)
    msh  = meshio.read(msh_path)
    msh.points *= 1e-3   # mm → m

    # ── Collect all tetrahedra and their tags ─────────────────────────────────
    tet_cells_list, tet_tags_list = [], []
    tri_cells_list, tri_tags_list = [], []

    # cell_data is indexed by block; we iterate in parallel with msh.cells
    gmsh_tags = msh.cell_data.get("gmsh:physical", [])

    for i, block in enumerate(msh.cells):
        tags = gmsh_tags[i] if i < len(gmsh_tags) else None
        if block.type == "tetra":
            tet_cells_list.append(block.data)
            if tags is not None:
                tet_tags_list.append(tags)
        elif block.type == "triangle":
            tri_cells_list.append(block.data)
            if tags is not None:
                tri_tags_list.append(tags)

    if not tet_cells_list:
        raise RuntimeError(
            "No tetrahedra in .msh — ensure Physical Volumes are defined."
        )

    tet_cells = np.vstack(tet_cells_list)
    tet_tags  = np.concatenate(tet_tags_list) if tet_tags_list else None

    # ── Write volume mesh ─────────────────────────────────────────────────────
    vol_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("tetra", tet_cells)],
        cell_data={"subdomain": [tet_tags]} if tet_tags is not None else {},
    )
    meshio.write(stem + ".xdmf", vol_mesh)
    print(f"[mesh] Written: {stem}.xdmf")

    # ── Write boundary facets ─────────────────────────────────────────────────
    if tri_cells_list:
        tri_cells = np.vstack(tri_cells_list)
        tri_tags  = np.concatenate(tri_tags_list) if tri_tags_list else None

        bnd_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("triangle", tri_cells)],
            cell_data={"boundaries": [tri_tags]} if tri_tags is not None else {},
        )
        meshio.write(stem + "_boundaries.xdmf", bnd_mesh)
        print(f"[mesh] Written: {stem}_boundaries.xdmf")
    else:
        print("[mesh] WARNING: no triangles found — boundary file not written")

    # ── Write subdomain file ──────────────────────────────────────────────────
    if tet_tags is not None:
        sub_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("tetra", tet_cells)],
            cell_data={"subdomains": [tet_tags]},
        )
        meshio.write(stem + "_subdomains.xdmf", sub_mesh)
        print(f"[mesh] Written: {stem}_subdomains.xdmf")

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_nodes = msh.points.shape[0]
    n_tets  = tet_cells.shape[0]
    n_tris  = tri_cells.shape[0] if tri_cells_list else 0
    print(f"\n[mesh] Statistics:")
    print(f"       Nodes          : {n_nodes:,}")
    print(f"       Tetrahedra     : {n_tets:,}")
    print(f"       Boundary faces : {n_tris:,}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  MESH INDEPENDENCE STUDY
# ════════════════════════════════════════════════════════════════════════════

def run_mesh_independence_study(
    base_cfg:          MeshConfig,
    refinement_levels: list[float] = [2.5, 1.5, 0.9, 0.5],
) -> None:
    """
    Generate four meshes at increasing refinement.
    Use the output DOF counts in your report to demonstrate convergence.

    Naming: {output_stem}_levelN.xdmf  (N = 0 coarsest → 3 finest)
    """
    print(f"\n[study] Mesh independence study — "
          f"lc_bulk sweep: {refinement_levels}")

    for k, lc in enumerate(refinement_levels):
        import copy
        cfg_k             = copy.deepcopy(base_cfg)
        cfg_k.lc_bulk     = lc
        cfg_k.lc_junction = lc * 0.4
        cfg_k.lc_plaque   = lc * 0.2
        cfg_k.output_stem = Path(str(base_cfg.output_stem) + f"_level{k}")
        print(f"\n  Level {k}: lc_bulk={lc:.2f} mm → {cfg_k.output_stem}")
        _run_single(cfg_k)


# ════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def _run_single(cfg: MeshConfig) -> None:
    """Run the full pipeline for one MeshConfig."""
    gmsh.initialize(["-nopopup"])
    gmsh.model.add("carotid")

    print(f"\n[build_mesh] output → {cfg.output_stem}")
    print(f"[build_mesh] Parent: D={cfg.D_inlet:.1f}mm  L={cfg.L_parent:.1f}mm")
    print(f"[build_mesh] Branch: D={cfg.D_outlet:.1f}mm  L={cfg.L_branch:.1f}mm"
          f"  φ={cfg.phi_deg:.1f}°")
    cfg.describe_plaques()

    fluid_vol = build_geometry(cfg)
    identify_and_tag_boundaries(fluid_vol, cfg)
    add_refinement_fields(cfg)
    generate_and_export(cfg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build carotid mesh (smooth bifurcation + sphere-cap plaques)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--R-sphere",   type=float, default=4.0,
                   help="Plaque sphere radius [mm]")
    p.add_argument("--intrusion",  type=float, default=1.2,
                   help="Plaque intrusion depth h [mm]  (0 = no plaque)")
    p.add_argument("--x-plaque",   type=float, default=40.0,
                   help="Axial position of plaque along branch [mm]")
    p.add_argument("--angle",      type=float, default=0.0,
                   help="Circumferential placement angle [°]  (0=top)")
    p.add_argument("--lc",         type=float, default=1.2,
                   help="Bulk element size [mm]")
    p.add_argument("--out",        type=str,   default="carotid",
                   help="Output file stem (no extension)")
    p.add_argument("--healthy",    action="store_true",
                   help="No plaques (healthy artery)")
    p.add_argument("--study",      action="store_true",
                   help="Run mesh independence study (4 levels)")
    p.add_argument("--folder",     type=str, default="default",
                   help="Subfolder in meshes/ to save outputs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.healthy or args.intrusion <= 0:
        plaques = []
    else:
        plaque = PlaqueConfig(
            R_sphere  = args.R_sphere,
            h         = args.intrusion,
            x_pos     = args.x_plaque,
            angle_deg = args.angle,
        )
        plaques = [plaque]

    cfg = MeshConfig(
        plaques_branch0 = plaques,
        plaques_branch1 = plaques,   # same plaque on both branches by default
        lc_bulk         = args.lc, #args.lc,
        lc_junction     = args.lc * 0.4, #args.lc * 0.4,
        lc_plaque       = args.lc * 0.2, #args.lc * 0.2,
        lc_wall         = args.lc * 0.3, #args.lc * 0.3,
        output_stem     = Path(f'meshes/{args.folder}/{args.out}'),
    )

    if args.study:
        run_mesh_independence_study(cfg)
    else:
        _run_single(cfg)

    print("\n[build_mesh] Complete.  Load in FEniCSx with:")
    print(f"    cfg.mesh_file = Path('meshes/{args.folder}/{args.out}.xdmf')")
    print( "    mesh, facet_tags, cell_tags = load_mesh(cfg)")


if __name__ == "__main__":
    main()
