# carotid-cfd

Pulsatile incompressible Navier–Stokes solver for a bifurcated carotid artery.
**FEniCSx (dolfinx)** — rigid walls — laminar flow — no FSI.

**4EM60 – Advanced Discretization Techniques**

---

## Layout

```
src/carotid_cfd/
    config.py      all parameters (SimulationConfig)
    tags.py        BoundaryTag / DomainTag integer constants
    geometry.py    load_mesh() — dolfinx XDMFFile API
    spaces.py      Taylor–Hood P2/P1 via basix.ufl
    boundary.py    BoundaryConditions class — dolfinx dirichletbc API
    weakform.py    NavierStokesForm — UFL residual + Jacobian
    postproc.py    FlowMonitor — assemble_scalar + MPI allreduce
    solver.py      NewtonSolver class + run_simulation()

examples/
    run_healthy.py
    run_stenosis_sweep.py

build_mesh.py      (your existing Gmsh script — unchanged)
```

---

## Install

```bash
# FEniCSx via Docker (recommended)
docker pull dolfinx/dolfinx:stable

# Or conda
conda create -n fenicsx -c conda-forge fenics-dolfinx
conda activate fenicsx

pip install -e .   # install carotid_cfd package
```

---

## Workflow

```bash
# 1. Generate mesh
python build_mesh.py --stenosis 0.5

# 2. Run
cd examples
python run_healthy.py
python run_stenosis_sweep.py
```

---

## dolfin → dolfinx cheat-sheet

| Topic | dolfin | dolfinx |
|---|---|---|
| Read mesh | `Mesh(); f.read(mesh)` | `f.read_mesh(name="Grid")` |
| Boundary tags | `MeshFunction` | `MeshTags` via `f.read_meshtags(mesh,...)` |
| Connectivity | implicit | `mesh.topology.create_connectivity(fdim,tdim)` |
| SubDomain | `class Inlet(SubDomain)` | `def inlet(x): return np.isclose(...)` |
| `near()` | built-in | `np.isclose(x[0], 0.0, atol=1e-8)` |
| Elements | `VectorElement(...)` | `basix.ufl.element(..., shape=(gdim,))` |
| FunctionSpace | `FunctionSpace(mesh, TH)` | `fem.functionspace(mesh, TH)` |
| collapse() | returns space | returns **(space, dof_map)** — use `V, _ = ...` |
| Constant | `Constant((0,0,0))` | `fem.Constant(mesh, np.zeros(3))` |
| DirichletBC | `DirichletBC(W.sub(0), val, bnd, tag)` | `fem.dirichletbc(val, dofs, W.sub(0))` |
| Locate DOFs | implicit | `fem.locate_dofs_topological((W.sub(0),V), fdim, facets)` |
| assemble | `assemble(form)` | `comm.allreduce(fem.assemble_scalar(fem.form(...)))` |
| XDMF write | `xdmf.write(u, t)` | `xdmf.write_function(u, t)` |
| DOLFIN_EPS | global constant | `1e-14` |
| NonlinearSolver | `NonlinearVariationalSolver` | **manual Newton** (see solver.py) |

---

## Newton solver — why manual?

`dolfinx.fem.petsc.NonlinearProblem` and `dolfinx.nls.petsc.NewtonSolver`
are **incompatible** in dolfinx 0.7–0.9:
- `NonlinearProblem` exposes `.A`, `.F`, `.x` (PETSc objects)
- `NewtonSolver` internally expects `.a`, `.L` (bilinear/linear UFL forms)

The solution is a hand-rolled Newton loop using the low-level assembly API:

```python
# Allocate ONCE per timestep (not inside the iteration loop!)
A  = fem.petsc.create_matrix(J_form)
b  = fem.petsc.create_vector(F_form)
du = b.copy()

# Per Newton iteration:
A.zeroEntries()
fem.petsc.assemble_matrix(A, J_form, bcs=bcs)
A.assemble()

with b.localForm() as b_loc: b_loc.set(0.0)
fem.petsc.assemble_vector(b, F_form)
fem.petsc.apply_lifting(b, [J_form], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

b.scale(-1.0)
ksp.solve(b, du)
du.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

up.x.petsc_vec.axpy(1.0, du)   # up += du
up.x.scatter_forward()
```

**Key rules to avoid segfaults:**
1. Allocate `A`, `b`, `du` **outside** the Newton iteration loop
2. Create the KSP solver **once per timestep**, destroy after
3. Always call `ghostUpdate` after assembly and after the solve
4. Always call `scatter_forward()` after updating the solution
