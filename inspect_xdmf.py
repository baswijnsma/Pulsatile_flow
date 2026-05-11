import meshio
import numpy as np
from pathlib import Path

def inspect(xdmf_path):
    print(f"\n--- Inspecting {xdmf_path} ---")

    with meshio.xdmf.TimeSeriesReader(xdmf_path) as reader:
        points, cells = reader.read_points_cells()
        print("Number of timesteps:", reader.num_steps)

        for i in range(reader.num_steps):
            t, point_data, _ = reader.read_data(i)
            print(f"\nTimestep {i}, time = {t}")

            for name, arr in point_data.items():
                print(f"  Field: {name}")
                print(f"    shape: {arr.shape}")
                print(f"    min  : {arr.min()}")
                print(f"    max  : {arr.max()}")
                print(f"    mean : {arr.mean()}")

if __name__ == "__main__":
    inspect("results_healthy/velocity.xdmf")
    inspect("results_healthy/pressure.xdmf")