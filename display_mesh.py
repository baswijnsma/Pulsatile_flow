import meshio
import numpy as np
import pyvista as pv

# -----------------------------
# FILES
# -----------------------------
vol_file = "meshes/healthy/carotid.xdmf"
bnd_file = "meshes/healthy/carotid_boundaries.xdmf"
sub_file = "meshes/healthy/carotid_subdomains.xdmf"

# -----------------------------
# LOAD VOLUME MESH (TETS)
# -----------------------------
vol_mesh = meshio.read(vol_file)

points = vol_mesh.points * 1e-3  # mm → m (keep consistent!)

tets = vol_mesh.get_cells_type("tetra")

grid = pv.UnstructuredGrid({10: tets}, points)

# -----------------------------
# LOAD SUBDOMAIN TAGS (volume labels)
# -----------------------------
sub_mesh = meshio.read(sub_file)

tet_tags = None
if "tetra" in sub_mesh.cell_data_dict:
    tet_tags = np.hstack(sub_mesh.cell_data_dict["tetra"])
    grid.cell_data["subdomain"] = tet_tags

# -----------------------------
# LOAD BOUNDARY SURFACE
# -----------------------------
bnd_mesh = meshio.read(bnd_file)

# meshio triangles (N x 3)
tris = bnd_mesh.get_cells_type("triangle")

# convert to VTK face format
faces = np.hstack(
    np.column_stack([
        np.full(len(tris), 3),  # each face has 3 points
        tris
    ])
).astype(np.int64)

surface = pv.PolyData(points, faces)
tri_tags = None
if "triangle" in bnd_mesh.cell_data_dict:
    tri_tags = np.hstack(bnd_mesh.cell_data_dict["triangle"])
    surface.cell_data["boundary"] = tri_tags

# -----------------------------
# PLOTTING
# -----------------------------
p = pv.Plotter(shape=(1, 2), window_size=(1200, 600))

# ---- Volume ----
p.subplot(0, 0)
p.add_text("Volume (Subdomains)", font_size=10)

p.add_mesh(
    grid,
    show_edges=False,
    opacity=0.15,
    scalars="subdomain" if tet_tags is not None else None,
    cmap="viridis"
)

# ---- Boundary ----
p.subplot(0, 1)
p.add_text("Boundary (Inlet/Outlet/Wall)", font_size=10)

p.add_mesh(
    surface,
    show_edges=True,
    scalars="boundary" if tri_tags is not None else None,
    cmap="plasma"
)

p.link_views()
p.show()