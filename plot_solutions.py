import meshio
import numpy as np
import pyvista as pv

# -----------------------------
# FILES
# -----------------------------
out_dir = "results_healthy"  # folder with velocity.xdmf and pressure.xdmf
vel_file = f"{out_dir}/velocity.xdmf"
pre_file = f"{out_dir}/pressure.xdmf"


# -----------------------------
# READ LAST STEP FROM XDMF
# -----------------------------
def read_last_step(xdmf_file):
    with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
        points, cells = reader.read_points_cells()
        nsteps = reader.num_steps
        _, point_data, _ = reader.read_data(nsteps - 1)
    return points, cells, point_data


def extract_tets(cells):
    for block in cells:
        if block.type == "tetra":
            return block.data
    raise RuntimeError("No tetra cells found.")


# -----------------------------
# LOAD DATA
# -----------------------------
print("Reading velocity...")
points, cells, vel_pd = read_last_step(vel_file)

print("Reading pressure...")
_, _, pre_pd = read_last_step(pre_file)

tets = extract_tets(cells)

# PyVista grid
grid = pv.UnstructuredGrid({10: tets}, points)

# -----------------------------
# ATTACH FIELDS
# -----------------------------
vel_name = list(vel_pd.keys())[0]
u = vel_pd[vel_name]
u_mag = np.linalg.norm(u, axis=1)

grid.point_data["velocity"] = u
grid.point_data["velocity_magnitude"] = u_mag

pre_name = list(pre_pd.keys())[0]
p = pre_pd[pre_name]
grid.point_data["pressure"] = p

print(f"|u| range: {u_mag.min():.3f} – {u_mag.max():.3f}")
print(f"p range:  {p.min():.3f} – {p.max():.3f}")

# -----------------------------
# STREAMLINES
# -----------------------------
print("Computing streamlines...")
streamlines = grid.streamlines(
    vectors="velocity",
    source_radius=0.003,
    n_points=400,
    max_time=1.0,
)

# -----------------------------
# PLOTTING (interactive window)
# -----------------------------
p = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

# ---- Velocity magnitude ----
p.subplot(0, 0)
p.add_text("Velocity magnitude", font_size=10)
p.add_mesh(
    grid,
    scalars="velocity_magnitude",
    cmap="plasma",
    show_edges=False,
)
p.view_isometric()

# ---- Pressure ----
p.subplot(0, 1)
p.add_text("Pressure", font_size=10)
p.add_mesh(
    grid,
    scalars="pressure",
    cmap="coolwarm",
    show_edges=False,
)
p.view_isometric()

# ---- Streamlines ----
p.subplot(0, 2)
p.add_text("Velocity streamlines", font_size=10)
p.add_mesh(
    grid,
    scalars="velocity_magnitude",
    cmap="plasma",
    opacity=0.3,
)
p.add_mesh(streamlines, color="black")
p.view_isometric()

p.link_views()
p.show()