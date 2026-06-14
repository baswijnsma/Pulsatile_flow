import meshio
import numpy as np
import pyvista as pv

# -----------------------------
# FILES
# -----------------------------
# vol_file = "meshes/healthy/carotid.xdmf"
# bnd_file = "meshes/healthy/carotid_boundaries.xdmf"
# sub_file = "meshes/healthy/carotid_subdomains.xdmf"

vol_file = "meshes/stenosis_30pct/carotid_30pct.xdmf"
bnd_file = "meshes/stenosis_30pct/carotid_30pct_boundaries.xdmf"
sub_file = "meshes/stenosis_30pct/carotid_30pct_subdomains.xdmf"


# -----------------------------
# LOAD VOLUME MESH (TETS)
# -----------------------------
vol_mesh = meshio.read(vol_file)

tet_tags = vol_mesh.cell_data_dict["subdomain"]["tetra"]

print("Unique volume tags:", np.unique(tet_tags))
print("Counts per tag:")
for t in np.unique(tet_tags):
    print(t, np.sum(tet_tags == t))

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

try:
    tri_tags = bnd_mesh.cell_data_dict["boundaries"]["triangle"]
    surface.cell_data["boundary"] = tri_tags
except KeyError:
    print("No boundary tags found.")

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
# -----------------------------
# BOUNDARY VISUALIZATION
# -----------------------------
p.subplot(0, 1)
p.add_text("Boundary Groups", font_size=10)


if tri_tags is not None:

    names = {
        1: "Inlet",
        2: "Outlet1",
        3: "Outlet2",
        4: "Wall",
    }

    colors = {
        1: "green",
        2: "red",
        3: "blue",
        4: "lightgray",
    }

    for tag in np.unique(tri_tags):

        mask = tri_tags == tag

        sub_surface = surface.extract_cells(
            np.where(mask)[0]
        )

        p.add_mesh(
            sub_surface,
            color=colors.get(tag, "white"),
            label=names.get(tag, f"Tag {tag}"),
            show_edges=True,
        )

    p.add_legend()

else:
    p.add_mesh(surface, show_edges=True)


for tag in np.unique(tri_tags):

    mask = tri_tags == tag

    sub_surface = surface.extract_cells(
        np.where(mask)[0]
    )

    center = sub_surface.center

    p.add_point_labels(
        [center],
        [names.get(tag, str(tag))],
        font_size=14,
        point_size=10,
    )
p.show()