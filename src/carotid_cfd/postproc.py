"""
carotid_cfd.postproc
====================
Flow rate, pressure, and flow-split extraction — dolfinx API.

Key dolfinx assembly change from dolfin
-----------------------------------------

dolfin (OLD):
    Q = assemble(dot(u, n) * ds(1))

dolfinx (NEW) — two steps:
    # Step 1: compile the form
    form = fem.form(dot(u, n) * ds(tag))
    # Step 2: assemble locally, then reduce over MPI ranks
    local_val = fem.assemble_scalar(form)
    global_val = mesh.comm.allreduce(local_val, op=MPI.SUM)

The MPI allreduce is required because in parallel each rank owns only
a portion of the mesh.  Without it, each rank returns a partial value.

Measure construction
--------------------
    ds = ufl.Measure("ds",
                     domain=mesh,
                     subdomain_data=facet_tags,
                     subdomain_id=tag)
    
Both  domain=  and  subdomain_data=  are required in dolfinx.
The  subdomain_id  can be set at construction time (one measure per tag)
or passed as an argument:  ds(tag)  — both patterns work.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import ufl
from ufl import dot, FacetNormal, Measure

from carotid_cfd.config import SimulationConfig
from carotid_cfd.tags import BoundaryTag

EPS = 1e-14


class FlowMonitor:
    """
    Computes and stores haemodynamic scalars at every timestep.

    Quantities
    ----------
    Q   [m³/s]  Volumetric flow rate  Q = ∫_Γ u·n dA
    P   [Pa]    Area-averaged pressure P̄ = (1/|Γ|) ∫_Γ p dA
    ΔP  [Pa]    Pressure drop  P_inlet − P_outlet
    α   [–]     Flow split ratio to outlet 1

    Usage
    -----
    >>> monitor = FlowMonitor(cfg, mesh, facet_tags)
    >>> rec = monitor.compute(t, up)
    >>> monitor.write_csv(output_dir / "flow_monitor.csv")
    >>> monitor.print_cycle_summary()
    """

    def __init__(
        self,
        cfg:        SimulationConfig,
        mesh:       dolfinx.mesh.Mesh,
        facet_tags,
    ):
        self.cfg     = cfg
        self.mesh    = mesh
        self.comm    = mesh.comm
        self.records: list[dict] = []

        n = FacetNormal(mesh)
        self._n = n

        # ── Integration measures — one per boundary tag ───────────────────────
        # subdomain_id is set here so we can write  self._ds_in  directly
        # without passing the tag every time.
        self._ds_in = Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              subdomain_id=BoundaryTag.INLET)
        self._ds_o1 = Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              subdomain_id=BoundaryTag.OUTLET_1)
        self._ds_o2 = Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              subdomain_id=BoundaryTag.OUTLET_2)

    def _assemble(self, integrand) -> float:
        """
        Assemble a scalar UFL integrand and sum over all MPI ranks.

        Pattern:
            local  = fem.assemble_scalar(fem.form(integrand))
            global = comm.allreduce(local, op=MPI.SUM)
        """
        local = fem.assemble_scalar(fem.form(integrand))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def compute(self, t: float, up: fem.Function) -> dict:
        """
        Extract scalar quantities from solution up at time t.

        Note: up.sub(0) and up.sub(1) return views (no copy).
        These can be used directly in UFL integrals.
        """
        n    = self._n
        u    = up.sub(0)   # velocity view
        p    = up.sub(1)   # pressure view
        one  = fem.Constant(self.mesh, dolfinx.default_scalar_type(1.0))

        # Flow rates  Q = ∫ u·n dA
        Q_in  = self._assemble(dot(u, n) * self._ds_in)
        Q_o1  = self._assemble(dot(u, n) * self._ds_o1)
        Q_o2  = self._assemble(dot(u, n) * self._ds_o2)

        # Boundary areas
        A_in  = self._assemble(one * self._ds_in)
        A_o1  = self._assemble(one * self._ds_o1)
        A_o2  = self._assemble(one * self._ds_o2)

        # Area-averaged pressures
        P_in  = self._assemble(p * self._ds_in)  / (A_in  + EPS)
        P_o1  = self._assemble(p * self._ds_o1) / (A_o1  + EPS)
        P_o2  = self._assemble(p * self._ds_o2) / (A_o2  + EPS)

        dP_1  = P_in - P_o1
        dP_2  = P_in - P_o2
        alpha = abs(Q_o1) / (abs(Q_o1) + abs(Q_o2) + EPS)

        record = dict(
            t=t,
            Q_in=Q_in, Q_out1=Q_o1, Q_out2=Q_o2,
            P_in=P_in, P_out1=P_o1, P_out2=P_o2,
            dP_1=dP_1, dP_2=dP_2,
            split=alpha,
        )
        self.records.append(record)
        return record

    def cycle_average(self, n_cycles_skip: int = 2) -> dict:
        """Time-average over records after n_cycles_skip * T_cycle."""
        T_skip = n_cycles_skip * self.cfg.waveform.T_cycle
        recs   = [r for r in self.records if r["t"] >= T_skip]
        if not recs:
            raise ValueError(
                f"No records after t = {T_skip:.2f} s.  "
                "Run more cycles or reduce n_cycles_skip."
            )
        keys = [k for k in recs[0] if k != "t"]
        return {k: float(np.mean([r[k] for r in recs])) for k in keys}

    def write_csv(self, path: Path) -> None:
        """Write all timestep records to CSV (rank-0 only)."""
        if self.comm.rank != 0 or not self.records:
            return
        keys = list(self.records[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[postproc] CSV written → {path}  ({len(self.records)} records)")

    def print_cycle_summary(self, n_cycles_skip: int = 2) -> None:
        try:
            avg = self.cycle_average(n_cycles_skip)
        except ValueError as e:
            print(f"[postproc] {e}")
            return

        mmHg = 133.322
        sep  = "─" * 50
        print(sep)
        print("  Cycle-averaged results")
        print(sep)
        print(f"  Q_in       = {avg['Q_in']*1e6:+.3f} ml/s")
        print(f"  Q_outlet_1 = {avg['Q_out1']*1e6:+.3f} ml/s")
        print(f"  Q_outlet_2 = {avg['Q_out2']*1e6:+.3f} ml/s")
        print(f"  Flow split = {avg['split']:.3f}  (fraction to outlet 1)")
        print(f"  <ΔP_1>     = {avg['dP_1']:.1f} Pa"
              f"  = {avg['dP_1']/mmHg:.2f} mmHg")
        print(f"  <ΔP_2>     = {avg['dP_2']:.1f} Pa"
              f"  = {avg['dP_2']/mmHg:.2f} mmHg")
        clinical = "SIGNIFICANT" if avg["dP_1"] > 1333 else "below threshold"
        print(f"  Clinical:    {clinical}  (threshold 10 mmHg)")
        print(sep)
