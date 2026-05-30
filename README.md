# Asset Articulator

An interactive tool for articulating 3-D mesh assets and exporting URDF files. 

Given a raw 3-D mesh (STL / PLY / OBJ), the tool lets you:

1. Define a **construction plane** over the mesh in a 3-D viewer.
2. Click to select a **cuboid** or **cylinder** region around a moving part (door, drawer, knob).
3. Pick a **hinge edge** (revolute) or **slider axis** (prismatic) and set joint limits.
4. Queue multiple articulations and **export a URDF** with properly split child meshes.

---

## Installation

> Recommended: use [uv](https://docs.astral.sh/uv/) with Python 3.10.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[develop]"
```

---

## Running the Tool

```bash
python scripts/cuboid_selector.py path/to/your/mesh.stl
```
Sample meshes can be found in **[docs/usage.md](docs/usage.md)**.
The interactive window will open. Use the right-hand panel to:

- Choose **Selection Mode**: Cuboid, Cylinder, or Cabinet (double-door).
- Adjust the **Construction Plane** orientation and position.
- Click twice on the blue plane to define a face, then tune depth with the slider.
- Click **Select Hinge (Revolute)** or **Select Slider (Prismatic)**, then click near a cuboid edge.
- Click **Add Door / Drawer** to queue the articulation.
- Click **Export URDF** when all parts are annotated.

For a full breakdown of every control, see **[docs/usage.md](docs/usage.md)**.

---

## Project Structure

```
src/asset_articulator/
├── assets/          # Joint and link data structures
├── geometry/        # Cuboid/cylinder clipping, door cutting, edge detection
├── io/              # Mesh I/O and URDF export
└── viewer/          # PyVista scene and overlay helpers

scripts/
└── cuboid_selector.py      # Main interactive application

tests/                      # Pytest unit tests
data/
├── input/                  # Input meshes
└── output/                 # Generated URDFs and split meshes
```

---

## Development

**Auto-format** (black + isort + docformatter):

```bash
./run_autoformat.sh
```

**Run all CI checks locally** (format, lint, type-check, tests):

```bash
./run_ci_checks.sh
```

**Run tests only:**

```bash
pytest tests/
```

---

## CI

GitHub Actions runs on every push:

| Check | Tool |
|---|---|
| Formatting | `black`, `isort` |
| Linting | `pylint` via `pytest-pylint` |
| Type checking | `mypy` |
| Unit tests | `pytest` |

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays and linear algebra |
| `trimesh` | Mesh loading and boolean operations |
| `pyvista` / `vtk` | 3-D rendering |
| `open3d` | Point cloud / mesh I/O helpers |
| `networkx` | Boundary loop graph for cap geometry |
| `matplotlib` | Utility plotting |
| `PyQt5` + `pyvistaqt` | GUI for the interactive selector |
