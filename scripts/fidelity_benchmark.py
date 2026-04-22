"""Compare PyBullet collision cost: AABB box vs. mesh (hull) vs. VHACD on a pile."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
SRC_STL = ROOT / "data/input/mug.stl"
OUT = ROOT / "data/output/fidelity_test"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = 0.0254          # inches -> m (mug ~12cm tall)
VHACD_RESOLUTION = 100_000
N_OBJECTS = 20
N_STEPS = 1000
N_RUNS = 3
SPAWN_SEED = 7  # Set to None for non-deterministic spawn orientations.

# Modes: "picker" = Qt picker window, "visual" = single variant in GUI, "benchmark" = headless timing
MODE = "picker"
VISUAL_VARIANT = "vhacd"  # used only when MODE == "visual"

ASSET = SRC_STL.stem


def random_unit_quaternion(rng: np.random.Generator) -> list[float]:
    """Sample a uniform random quaternion."""
    u1, u2, u3 = rng.random(3)
    return [
        float(np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)),
        float(np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)),
        float(np.sqrt(u1) * np.sin(2.0 * np.pi * u3)),
        float(np.sqrt(u1) * np.cos(2.0 * np.pi * u3)),
    ]


def preprocess() -> tuple[Path, Path, Path, np.ndarray]:
    m = trimesh.load(SRC_STL, process=False, force="mesh")
    assert isinstance(m, trimesh.Trimesh)
    m.apply_scale(SCALE)
    m.apply_translation(-m.centroid)

    mesh_stl = OUT / f"{ASSET}_mesh.stl"
    mesh_obj = OUT / f"{ASSET}_mesh.obj"
    m.export(mesh_stl)
    m.export(mesh_obj)
    print(f"Mesh: {len(m.faces)} faces -> {mesh_stl}")

    hull_stl = OUT / f"{ASSET}_hull.stl"
    hull = m.convex_hull
    hull.export(hull_stl)
    print(f"Hull: {len(hull.faces)} faces -> {hull_stl}")

    # VHACD decomposition (cached)
    vhacd_obj = OUT / f"{ASSET}_vhacd.obj"
    vhacd_log = OUT / f"{ASSET}_vhacd.log"
    if not vhacd_obj.exists():
        print(f"Running VHACD (resolution={VHACD_RESOLUTION})... this takes a bit.")
        cid = p.connect(p.DIRECT)
        p.vhacd(str(mesh_obj), str(vhacd_obj), str(vhacd_log),
                resolution=VHACD_RESOLUTION)
        p.disconnect(cid)
        print(f"VHACD done -> {vhacd_obj}")
    else:
        print(f"VHACD cached -> {vhacd_obj}")

    return mesh_stl, vhacd_obj, hull_stl, m.extents


def write_urdf(path: Path, collision_xml: str, visual_xml: str) -> None:
    urdf = f"""<?xml version="1.0"?>
<robot name="{ASSET}">
  <link name="base">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="5e-4" ixy="0" ixz="0" iyy="5e-4" iyz="0" izz="5e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      {visual_xml}
      <material name="orange"><color rgba="0.9 0.5 0.2 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      {collision_xml}
    </collision>
  </link>
</robot>
"""
    path.write_text(urdf)


def build_urdfs(
    mesh_stl: Path, vhacd_obj: Path, hull_stl: Path, extents: np.ndarray
) -> dict[str, Path]:
    box_xml   = f'<geometry><box size="{extents[0]:.5f} {extents[1]:.5f} {extents[2]:.5f}"/></geometry>'
    mesh_xml  = f'<geometry><mesh filename="{mesh_stl.name}"/></geometry>'
    hull_xml  = f'<geometry><mesh filename="{hull_stl.name}"/></geometry>'
    vhacd_xml = f'<geometry><mesh filename="{vhacd_obj.name}"/></geometry>'

    urdfs = {
        "BOX (AABB)":         OUT / f"{ASSET}_box.urdf",
        "MESH (hull)":        OUT / f"{ASSET}_mesh.urdf",
        "VHACD":              OUT / f"{ASSET}_vhacd.urdf",
        "BOX [collision]":    OUT / f"{ASSET}_box_col.urdf",
        "MESH [collision]":   OUT / f"{ASSET}_mesh_col.urdf",
        "VHACD [collision]":  OUT / f"{ASSET}_vhacd_col.urdf",
    }
    # Physics-accurate visual (what the user would normally see)
    write_urdf(urdfs["BOX (AABB)"], box_xml, box_xml)
    write_urdf(urdfs["MESH (hull)"], mesh_xml, mesh_xml)
    write_urdf(urdfs["VHACD"], vhacd_xml, mesh_xml)
    # Collision-geometry view: visual = true collision shape Bullet uses
    write_urdf(urdfs["BOX [collision]"], box_xml, box_xml)
    write_urdf(urdfs["MESH [collision]"], mesh_xml, hull_xml)  # show actual hull
    write_urdf(urdfs["VHACD [collision]"], vhacd_xml, vhacd_xml)
    return urdfs


def count_obj_parts(obj_path: Path) -> tuple[int, int]:
    """Return (num_groups, total_verts) in a multi-group OBJ. VHACD output uses 'o' per convex part."""
    if not obj_path.exists():
        return 0, 0
    n_groups = 0
    n_verts = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith(("o ", "g ")):
                n_groups += 1
            elif line.startswith("v "):
                n_verts += 1
    return max(n_groups, 1), n_verts


def inspect_collision(urdf_path: Path, vhacd_obj: Path | None = None) -> None:
    """Print what Bullet stores as collision geometry, plus OBJ-group count for VHACD."""
    cid = p.connect(p.DIRECT)
    bid = p.loadURDF(str(urdf_path))
    shapes = p.getCollisionShapeData(bid, -1)
    print(f"  [{urdf_path.name}] getCollisionShapeData reports {len(shapes)} shape(s)")
    for i, s in enumerate(shapes):
        geom_type = s[2]
        try:
            nverts, _ = p.getMeshData(bid, -1, collisionShapeIndex=i)
        except p.error:
            nverts = None
        print(f"    shape {i}: geom_type={geom_type}  verts={nverts}")
    p.disconnect(cid)

    if vhacd_obj is not None and vhacd_obj.exists():
        parts, verts = count_obj_parts(vhacd_obj)
        print(f"    VHACD OBJ groundtruth: {parts} convex parts, {verts} total verts")


def run_sim(urdf_path: Path, extents: np.ndarray, gui: bool = False) -> dict:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    rng = np.random.default_rng(SPAWN_SEED)
    if gui:
        p.setAdditionalSearchPath(str(OUT))
        p.resetDebugVisualizerCamera(
            cameraDistance=max(extents) * 6, cameraYaw=45, cameraPitch=-75,
            cameraTargetPosition=[0, 0, 0],
        )

    # Static bin: floor + 4 walls
    half = max(extents) * 3.0
    floor = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, floor)
    wall_h = max(extents) * 4.0
    wall_t = 0.01
    for (cx, cy, sx, sy) in [
        ( half, 0, wall_t, half),
        (-half, 0, wall_t, half),
        (0,  half, half, wall_t),
        (0, -half, half, wall_t),
    ]:
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, wall_h])
        p.createMultiBody(0, shape, basePosition=[cx, cy, wall_h])

    t_load = time.perf_counter()
    ids = []
    side = int(np.ceil(np.sqrt(N_OBJECTS)))
    spacing = max(extents) * 1.2
    jitter = spacing * 0.35
    for i in range(N_OBJECTS):
        gx, gy = i % side, i // side
        x = (gx - side / 2) * spacing + rng.uniform(-jitter, jitter)
        y = (gy - side / 2) * spacing + rng.uniform(-jitter, jitter)
        z = 0.45 + rng.uniform(0.0, 0.4) + 0.05 * (i % 3)
        orn = random_unit_quaternion(rng)
        bid = p.loadURDF(str(urdf_path), basePosition=[x, y, z], baseOrientation=orn)
        ids.append(bid)
    load_time = time.perf_counter() - t_load

    t_step = time.perf_counter()
    step_limit = 10**9 if gui else N_STEPS
    for step in range(step_limit):
        try:
            p.stepSimulation()
        except p.error:
            break
        if gui:
            if not p.getConnectionInfo(cid).get("isConnected", 0):
                break
            time.sleep(1.0 / 240.0)
            if step % 240 == 0:
                pos0, _ = p.getBasePositionAndOrientation(ids[0])
                print(f"  t={step/240:.1f}s  obj[0] z={pos0[2]:+.3f}m")
    step_time = time.perf_counter() - t_step

    try:
        n_contacts = sum(len(p.getContactPoints(b)) for b in ids)
    except p.error:
        n_contacts = -1
    try:
        p.disconnect(cid)
    except p.error:
        pass
    return {
        "load_s": load_time,
        "step_s": step_time,
        "per_step_ms": 1000 * step_time / N_STEPS,
        "steps_per_s": N_STEPS / step_time,
        "final_contacts": n_contacts,
    }


def run_benchmark(urdfs: dict[str, Path], vhacd_obj: Path, extents: np.ndarray) -> None:
    print("\n--- Collision shape inspection ---")
    for name, urdf in urdfs.items():
        print(f"\n{name}:")
        inspect_collision(urdf, vhacd_obj if name == "VHACD" else None)
    print()

    results: dict[str, float] = {}
    for name, urdf in urdfs.items():
        if "[collision]" in name:
            continue
        runs = []
        for r in range(N_RUNS):
            res = run_sim(urdf, extents, gui=False)
            runs.append(res)
            print(f"[{name}] run {r+1}: {res['per_step_ms']:.3f} ms/step, "
                  f"{res['steps_per_s']:.1f} steps/s, load {res['load_s']:.2f}s, "
                  f"contacts@end={res['final_contacts']}")
        avg = float(np.mean([r["per_step_ms"] for r in runs]))
        results[name] = avg
        print(f"[{name}] avg per-step: {avg:.3f} ms\n")

    print("=" * 50)
    for name, ms in results.items():
        print(f"  {name:22s}  {ms:7.3f} ms/step")
    baseline = results["BOX (AABB)"]
    print()
    for name, ms in results.items():
        if name != "BOX (AABB)":
            print(f"  {name} is {ms / baseline:.2f}x slower than BOX")


def launch_picker(urdfs: dict[str, Path], vhacd_obj: Path, extents: np.ndarray) -> None:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    )

    app = QApplication.instance() or QApplication([])
    win = QWidget()
    win.setWindowTitle("Fidelity Benchmark")
    lay = QVBoxLayout(win)
    lay.addWidget(QLabel("Visualize one variant (close the PyBullet window to return):"))
    def _watch(u: Path) -> None:
        run_sim(u, extents, gui=True)

    for name, urdf in urdfs.items():
        btn = QPushButton(f"Watch {name}")
        btn.clicked.connect(lambda _=False, u=urdf: _watch(u))
        lay.addWidget(btn)
    bench_btn = QPushButton("Run Benchmark (all tiers, headless, timed)")
    bench_btn.clicked.connect(lambda: run_benchmark(urdfs, vhacd_obj, extents))
    lay.addWidget(bench_btn)
    quit_btn = QPushButton("Quit")
    quit_btn.clicked.connect(app.quit)
    lay.addWidget(quit_btn)
    win.resize(320, 0)
    win.show()
    app.exec_()


def main() -> None:
    mesh_stl, vhacd_obj, hull_stl, extents = preprocess()
    urdfs = build_urdfs(mesh_stl, vhacd_obj, hull_stl, extents)

    if MODE == "visual":
        key = {"box": "BOX (AABB)", "mesh": "MESH (hull)", "vhacd": "VHACD"}[VISUAL_VARIANT]
        print(f"visual mode — running {key} in GUI (no timing).")
        run_sim(urdfs[key], extents, gui=True)
    elif MODE == "picker":
        launch_picker(urdfs, vhacd_obj, extents)
    else:  # "benchmark"
        run_benchmark(urdfs, vhacd_obj, extents)


if __name__ == "__main__":
    main()
