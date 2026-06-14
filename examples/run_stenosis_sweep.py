"""
examples/run_stenosis_sweep.py
==============================
Sweep stenosis severity δ ∈ {0.0, 0.3, 0.5, 0.7}.

    # Generate meshes first:
    for d in 0.0 0.3 0.5 0.7; do
        python ../build_mesh.py --stenosis $d \
            --out carotid_$(python3 -c "print(int($d*100))")pct
    done
    python run_stenosis_sweep.py
"""
from __future__ import annotations
import csv
from pathlib import Path
from carotid_cfd import (SimulationConfig, GeometryParameters,
                          SolverConfig, run_simulation)

DELTAS = [ 0.0]


def main():
    mmHg    = 133.322
    summary = []

    for delta in DELTAS:
        pct = int(round(delta * 100))
        mesh_file = Path(f"meshes/stenosis_{pct:02d}pct/carotid_{pct:02d}pct.xdmf")

        if not mesh_file.exists():
            print(f"[skip] {mesh_file} not found")
            continue

        cfg = SimulationConfig(
            geometry   = GeometryParameters(stenosis_severity=delta),
            solver     = SolverConfig(dt=1e-3, t_end=3.0, output_interval=100),
            mesh_file  = mesh_file,
            output_dir = Path(f"results_{pct:02d}pct"),
        )
        monitor = run_simulation(cfg)
        avg     = monitor.cycle_average(n_cycles_skip=2)

        row = dict(
            delta=delta,
            area_pct   = round((1-(1-delta)**2)*100, 1),
            dP_1_mmHg  = round(avg["dP_1"] / mmHg, 2),
            dP_2_mmHg  = round(avg["dP_2"] / mmHg, 2),
            split      = round(avg["split"], 4),
            Q1_mlps    = round(avg["Q_out1"]*1e6, 3),
            Q2_mlps    = round(avg["Q_out2"]*1e6, 3),
        )
        summary.append(row)
        print(f"  δ={delta:.1f}  ΔP₁={row['dP_1_mmHg']:.2f} mmHg"
              f"  α={row['split']:.3f}")

    if summary:
        p = Path("sweep_summary.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        print(f"\nSummary → {p}")

if __name__ == "__main__":
    main()
