from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pyvista as pv
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.mesh_ops import split_mesh_by_cuboid


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation formula."""
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


@dataclass
class FaceSelection:
    p0_uv: np.ndarray | None = None
    p1_uv: np.ndarray | None = None


class CuboidSelectorApp:
    """Viewer for selecting a cuboid by:
    1) drawing a rectangle on a rotatable construction plane
    2) extruding along the plane normal
    3) optionally moving only the camera to view the plane head-on
    """

    def __init__(self, mesh_path: str | Path) -> None:
        self.mesh_path = Path(mesh_path)
        self.mesh_tm = trimesh.load_mesh(self.mesh_path, process=False)
        if not isinstance(self.mesh_tm, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(self.mesh_tm)}")

        self.mesh_pv = pv.wrap(self.mesh_tm)

        bounds = self.mesh_tm.bounds
        self.bounds_min = bounds[0].astype(float)
        self.bounds_max = bounds[1].astype(float)
        self.scene_center = 0.5 * (self.bounds_min + self.bounds_max)
        self.scene_extents = self.bounds_max - self.bounds_min
        self.scene_diag = float(np.linalg.norm(self.scene_extents))

        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.show_grid()

        # Construction plane state.
        self.plane_origin = np.array(
            [
                self.scene_center[0],
                self.bounds_max[1] - 0.02 * self.scene_extents[1],
                self.scene_center[2],
            ],
            dtype=float,
        )
        self.plane_u = np.array([1.0, 0.0, 0.0], dtype=float)
        self.plane_v = np.array([0.0, 0.0, 1.0], dtype=float)
        self.plane_size_u = max(1e-3, 1.5 * float(self.scene_extents[0]))
        self.plane_size_v = max(1e-3, 1.5 * float(self.scene_extents[2]))

        self.depth = max(0.05, 0.10 * float(max(self.scene_extents[1], 1e-3)))
        self.extrude_sign = -1.0

        self.face = FaceSelection()
        self.current_cuboid: OrientedCuboid | None = None

        self.mesh_actor = None
        self.plane_actor = None
        self.face_actor = None
        self.box_actor = None
        self.text_actor = None

        self._build_scene()
        self._register_callbacks()

    @property
    def plane_n(self) -> np.ndarray:
        return normalize(np.cross(self.plane_u, self.plane_v))

    def _build_scene(self) -> None:
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_pv,
            color="lightgray",
            opacity=0.8,
            show_edges=False,
            name="object_mesh",
            pickable=False,
        )
        self._update_plane()
        self._update_text()

    def _register_callbacks(self) -> None:
        self.plotter.enable_surface_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_point=False,
            show_message=True,
        )

        self.plotter.add_key_event("r", self._reset_face)

        self.plotter.add_key_event("j", self._move_plane_backward)
        self.plotter.add_key_event("k", self._move_plane_forward)

        self.plotter.add_key_event("n", self._decrease_depth)
        self.plotter.add_key_event("m", self._increase_depth)
        self.plotter.add_key_event("d", self._flip_extrusion_direction)

        # 1-degree plane rotation
        self.plotter.add_key_event("a", self._rotate_plane_yaw_neg)
        self.plotter.add_key_event("f", self._rotate_plane_yaw_pos)
        self.plotter.add_key_event("q", self._rotate_plane_pitch_neg)
        self.plotter.add_key_event("e", self._rotate_plane_pitch_pos)

        # Camera-only alignment
        self.plotter.add_key_event("p", lambda: self._view_plane_head_on(side=1.0))
        self.plotter.add_key_event("t", lambda: self._view_plane_head_on(side=-1.0))
        self.plotter.add_key_event("c", self._print_current_cuboid)
        self.plotter.add_key_event("s", self._save_split_meshes)

    def _make_plane_mesh(self) -> pv.PolyData:
        half_u = 0.5 * self.plane_size_u
        half_v = 0.5 * self.plane_size_v

        corners = np.array(
            [
                self.plane_origin - half_u * self.plane_u - half_v * self.plane_v,
                self.plane_origin + half_u * self.plane_u - half_v * self.plane_v,
                self.plane_origin + half_u * self.plane_u + half_v * self.plane_v,
                self.plane_origin - half_u * self.plane_u + half_v * self.plane_v,
            ],
            dtype=float,
        )
        faces = np.hstack([[4, 0, 1, 2, 3]])
        return pv.PolyData(corners, faces)

    def _update_plane(self) -> None:
        plane = self._make_plane_mesh()

        if self.plane_actor is not None:
            self.plotter.remove_actor(self.plane_actor)

        self.plane_actor = self.plotter.add_mesh(
            plane,
            color="deepskyblue",
            opacity=0.25,
            show_edges=True,
            name="construction_plane",
            pickable=True,
        )

        self._update_face_preview()
        self._update_cuboid_preview()
        self._update_text()
        self.plotter.render()

    def _update_text(self) -> None:
        n = self.plane_n
        help_text = (
            "Cuboid selector\n\n"
            "Click twice on the blue construction plane to define a face.\n\n"
            "Keys:\n"
            "  r : reset face selection\n"
            "  j/k : move plane along normal\n"
            "  n/m : decrease/increase depth\n"
            "  d : flip extrusion direction\n"
            "  a/f : rotate plane yaw -/+ 1 deg (about world Z)\n"
            "  q/e : rotate plane pitch -/+ 1 deg (about plane U)\n"
            "  p : view current plane head-on\n"
            "  c : print current cuboid\n"
            "  s : split mesh and save STL files\n\n"
            f"depth = {self.depth:.4f}\n"
            f"dir   = {'-normal' if self.extrude_sign < 0 else '+normal'}\n"
            f"n     = [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]"
        )

        if self.text_actor is not None:
            self.plotter.remove_actor(self.text_actor)
        self.text_actor = self.plotter.add_text(
            help_text,
            position="upper_left",
            font_size=10,
        )

    def _on_pick(self, point: np.ndarray) -> None:
        try:
            p_world = np.array(point, dtype=float)
            uv = self._world_to_plane_uv(p_world)

            if self.face.p0_uv is None:
                self.face.p0_uv = uv
                print(f"[pick] p0_uv = {self.face.p0_uv}")
            elif self.face.p1_uv is None:
                self.face.p1_uv = uv
                print(f"[pick] p1_uv = {self.face.p1_uv}")
            else:
                self.face.p0_uv = uv
                self.face.p1_uv = None
                print(f"[pick] restarting face selection, new p0_uv = {self.face.p0_uv}")

            self._update_face_preview()
            self._update_cuboid_preview()
            self.plotter.render()
        except Exception as exc:
            print(f"[error] pick failed: {exc}")

    def _world_to_plane_uv(self, p_world: np.ndarray) -> np.ndarray:
        delta = p_world - self.plane_origin
        u = np.dot(delta, self.plane_u)
        v = np.dot(delta, self.plane_v)
        return np.array([u, v], dtype=float)

    def _plane_uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        return self.plane_origin + uv[0] * self.plane_u + uv[1] * self.plane_v

    def _update_face_preview(self) -> None:
        if self.face_actor is not None:
            self.plotter.remove_actor(self.face_actor)
            self.face_actor = None

        if self.face.p0_uv is None or self.face.p1_uv is None:
            return

        u0, v0 = self.face.p0_uv
        u1, v1 = self.face.p1_uv

        umin, umax = sorted([u0, u1])
        vmin, vmax = sorted([v0, v1])

        if (umax - umin) <= 1e-9 or (vmax - vmin) <= 1e-9:
            return

        corners_uv = np.array(
            [
                [umin, vmin],
                [umax, vmin],
                [umax, vmax],
                [umin, vmax],
            ],
            dtype=float,
        )
        corners_world = np.array([self._plane_uv_to_world(uv) for uv in corners_uv])

        faces = np.hstack([[4, 0, 1, 2, 3]])
        quad = pv.PolyData(corners_world, faces)

        self.face_actor = self.plotter.add_mesh(
            quad,
            color="orange",
            opacity=0.45,
            show_edges=True,
            line_width=3,
            name="face_preview",
        )

    def _update_cuboid_preview(self) -> None:
        if self.box_actor is not None:
            self.plotter.remove_actor(self.box_actor)
            self.box_actor = None

        self.current_cuboid = None

        if self.face.p0_uv is None or self.face.p1_uv is None:
            return

        u0, v0 = self.face.p0_uv
        u1, v1 = self.face.p1_uv

        umin, umax = sorted([u0, u1])
        vmin, vmax = sorted([v0, v1])

        center_uv = np.array([0.5 * (umin + umax), 0.5 * (vmin + vmax)], dtype=float)
        face_center = self._plane_uv_to_world(center_uv)

        half_u = 0.5 * (umax - umin)
        half_v = 0.5 * (vmax - vmin)
        half_n = 0.5 * self.depth

        if half_u <= 1e-9 or half_v <= 1e-9 or half_n <= 1e-9:
            return

        n = self.plane_n
        cuboid_center = face_center + self.extrude_sign * half_n * n

        rotation = np.column_stack([self.plane_u, self.plane_v, n])

        cuboid = OrientedCuboid(
            center=cuboid_center,
            rotation=rotation,
            extents=np.array([half_u, half_v, half_n], dtype=float),
        )
        self.current_cuboid = cuboid

        corners_local = np.array(
            [
                [-half_u, -half_v, -half_n],
                [ half_u, -half_v, -half_n],
                [ half_u,  half_v, -half_n],
                [-half_u,  half_v, -half_n],
                [-half_u, -half_v,  half_n],
                [ half_u, -half_v,  half_n],
                [ half_u,  half_v,  half_n],
                [-half_u,  half_v,  half_n],
            ],
            dtype=float,
        )
        corners_world = cuboid.local_to_world(corners_local)

        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        line_cells: list[int] = []
        for a, b in edge_pairs:
            line_cells.extend([2, a, b])

        wire = pv.PolyData()
        wire.points = corners_world
        wire.lines = np.array(line_cells, dtype=np.int32)

        self.box_actor = self.plotter.add_mesh(
            wire,
            color="red",
            line_width=3,
            name="cuboid_preview",
        )

    def _reset_face(self) -> None:
        print("[info] reset face selection")
        self.face = FaceSelection()
        self.current_cuboid = None

        if self.face_actor is not None:
            self.plotter.remove_actor(self.face_actor)
            self.face_actor = None
        if self.box_actor is not None:
            self.plotter.remove_actor(self.box_actor)
            self.box_actor = None

        self.plotter.render()

    def _move_plane_backward(self) -> None:
        try:
            step = 0.005 * self.scene_diag
            self.plane_origin = self.plane_origin - step * self.plane_n
            print(f"[info] moved plane backward to {self.plane_origin}")
            self._update_plane()
        except Exception as exc:
            print(f"[error] move plane backward failed: {exc}")

    def _move_plane_forward(self) -> None:
        try:
            step = 0.02 * self.scene_diag
            self.plane_origin = self.plane_origin + step * self.plane_n
            print(f"[info] moved plane forward to {self.plane_origin}")
            self._update_plane()
        except Exception as exc:
            print(f"[error] move plane forward failed: {exc}")

    def _decrease_depth(self) -> None:
        try:
            step = 0.02 * self.scene_diag
            self.depth = max(step, self.depth - step)
            print(f"[info] depth = {self.depth:.4f}")
            self._update_cuboid_preview()
            self._update_text()
            self.plotter.render()
        except Exception as exc:
            print(f"[error] decrease depth failed: {exc}")

    def _increase_depth(self) -> None:
        try:
            step = 0.02 * self.scene_diag
            self.depth += step
            print(f"[info] depth = {self.depth:.4f}")
            self._update_cuboid_preview()
            self._update_text()
            self.plotter.render()
        except Exception as exc:
            print(f"[error] increase depth failed: {exc}")

    def _flip_extrusion_direction(self) -> None:
        try:
            self.extrude_sign *= -1.0
            print("[info] extrusion direction flipped")
            self._update_cuboid_preview()
            self._update_text()
            self.plotter.render()
        except Exception as exc:
            print(f"[error] flip extrusion direction failed: {exc}")

    def _rotate_plane_yaw_neg(self) -> None:
        self._rotate_plane_about_axis(np.array([0.0, 0.0, 1.0]), -1.0)

    def _rotate_plane_yaw_pos(self) -> None:
        self._rotate_plane_about_axis(np.array([0.0, 0.0, 1.0]), 1.0)

    def _rotate_plane_pitch_neg(self) -> None:
        self._rotate_plane_about_axis(self.plane_u, -1.0)

    def _rotate_plane_pitch_pos(self) -> None:
        self._rotate_plane_about_axis(self.plane_u, 1.0)

    def _rotate_plane_about_axis(self, axis_world: np.ndarray, angle_deg: float) -> None:
        try:
            angle_rad = np.deg2rad(angle_deg)
            R = rotation_matrix(axis_world, angle_rad)

            new_u = R @ self.plane_u
            new_v = R @ self.plane_v

            new_u = normalize(new_u)
            new_v = new_v - np.dot(new_v, new_u) * new_u
            new_v = normalize(new_v)

            self.plane_u = new_u
            self.plane_v = new_v

            print(f"[info] rotated plane by {angle_deg:.1f} deg about axis {axis_world}")
            self._update_plane()
        except Exception as exc:
            print(f"[error] rotation failed: {exc}")

    def _view_plane_head_on(self, side) -> None:
        """Move only the camera so the current plane is viewed head-on.

        This does NOT change the plane position or orientation.
        """
        try:
            camera = self.plotter.camera

            n = self.plane_n
            up = normalize(self.plane_v)

            current_position = np.array(camera.position, dtype=float)
            dist = np.linalg.norm(current_position - self.plane_origin)

            if dist < 1e-6:
                dist = max(1.0, 1.2 * self.scene_diag)

            new_position = self.plane_origin + side* dist * n

            camera.position = tuple(new_position)
            camera.focal_point = tuple(self.plane_origin)
            camera.up = tuple(up)

            self.plotter.render()
            print("[info] camera aligned to view current plane head-on")
        except Exception as exc:
            print(f"[error] view plane head-on failed: {exc}")

    def _print_current_cuboid(self) -> None:
        if self.current_cuboid is None:
            print("[warn] no cuboid defined yet")
            return

        print("\nCurrent cuboid:")
        print(f"center = np.array({self.current_cuboid.center.tolist()})")
        print(f"rotation = np.array({self.current_cuboid.rotation.tolist()})")
        print(f"extents = np.array({self.current_cuboid.extents.tolist()})\n")

    def _save_split_meshes(self) -> None:
        if self.current_cuboid is None:
            print("[warn] no cuboid defined yet")
            return

        try:
            result = split_mesh_by_cuboid(self.mesh_tm, self.current_cuboid)

            output_dir = Path("data/output/split_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            inside_path = output_dir / "inside_clip.stl"
            outside_path = output_dir / "outside_clip.stl"

            result.inside_mesh.export(inside_path)
            result.outside_mesh.export(outside_path)

            print(f"[save] inside  -> {inside_path}")
            print(f"[save] outside -> {outside_path}")
            print(f"[save] inside faces  = {len(result.inside_mesh.faces)}")
            print(f"[save] outside faces = {len(result.outside_mesh.faces)}")
        except Exception as exc:
            print(f"[error] saving split meshes failed: {exc}")

    def run(self) -> None:
        self.plotter.show()


def main() -> None:
    app = CuboidSelectorApp("data/input/object.stl")
    app.run()


if __name__ == "__main__":
    main()