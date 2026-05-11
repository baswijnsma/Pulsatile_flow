"""
examples/run_healthy.py
=======================
Baseline: healthy artery, no stenosis.

    pip install -e ..
    python build_mesh.py --stenosis 0.0 --out carotid_healthy
    python run_healthy.py
"""
from pathlib import Path
from carotid_cfd import SimulationConfig, GeometryParameters, SolverConfig, run_simulation

def main():
    cfg = SimulationConfig(
        geometry   = GeometryParameters(stenosis_severity=0.0),
        solver     = SolverConfig(dt=5e-2, t_end=3.0, output_interval=20),
        mesh_file  = Path("meshes/healthy/carotid.xdmf"),
        output_dir = Path("results_healthy"),

    )
    monitor = run_simulation(cfg)
    monitor.print_cycle_summary(n_cycles_skip=2)

if __name__ == "__main__":
    main()
