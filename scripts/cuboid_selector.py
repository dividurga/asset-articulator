from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pyvista as pv
import pyvistaqt
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QSlider, QDoubleSpinBox, QPushButton, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal

from asset_articulator.assets.joints import JointLimits
from asset_articulator.geometry.clip import split_mesh_by_cuboid_clip
from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.edge import Edge
from asset_articulator.geometry.door import cut_cuboid_with_surface
from asset_articulator.io.urdf_export import export_to_urdf


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation formula."""
    axis = normalize(axis)
    x, y, z = axis
    c, s, C = np.cos(angle_rad), np.sin(angle_rad), 1.0 - np.cos(angle_rad)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Reusable slider + spinbox widget
# ---------------------------------------------------------------------------

class SliderSpinBox(QWidget):
    """Labeled QSlider + QDoubleSpinBox that stay in sync.

    Emits valueChanged(float) whenever the value changes from either widget.
    set_value() does NOT emit (use it for programmatic resets).
    """
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        init_val: float,
        decimals: int = 3,
        steps: int = 1000,
        parent=None,
    ):
        super().__init__(parent)
        self._steps = steps

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._lbl = QLabel(label)
        layout.addWidget(self._lbl)

        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(decimals)
        row.addWidget(self.slider, stretch=3)
        row.addWidget(self.spinbox, stretch=1)
        layout.addLayout(row)

        self.set_range(min_val, max_val)
        self.set_value(init_val)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)

    # ------------------------------------------------------------------
    def set_range(self, min_val: float, max_val: float) -> None:
        self._min = float(min_val)
        self._max = float(max_val)
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.spinbox.setRange(self._min, self._max)
        self.spinbox.setSingleStep((self._max - self._min) / self._steps)
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)

    def set_label(self, text: str) -> None:
        self._lbl.setText(text)

    def set_value(self, val: float) -> None:
        """Set value silently (no valueChanged emitted)."""
        clamped = float(np.clip(val, self._min, self._max))
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.spinbox.setValue(clamped)
        self.slider.setValue(self._val_to_slider(clamped))
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)

    def value(self) -> float:
        return self.spinbox.value()

    # ------------------------------------------------------------------
    def _val_to_slider(self, val: float) -> int:
        span = self._max - self._min
        if span < 1e-12:
            return 0
        frac = (val - self._min) / span
        return int(round(float(np.clip(frac * self._steps, 0, self._steps))))

    def _slider_to_val(self, pos: int) -> float:
        return self._min + (pos / self._steps) * (self._max - self._min)

    def _slider_changed(self, pos: int) -> None:
        val = self._slider_to_val(pos)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(val)

    def _spinbox_changed(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._val_to_slider(val))
        self.slider.blockSignals(False)
        self.valueChanged.emit(val)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FaceSelection:
    p0_uv: np.ndarray | None = None
    p1_uv: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CuboidSelectorApp(QMainWindow):
    def __init__(self, mesh_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("Cuboid Selector")

        # Load mesh --------------------------------------------------------
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

        # Plane state (cumulative angles + absolute offset along normal) ---
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.plane_offset = 0.0
        self.plane_origin_base = np.array([
            self.scene_center[0],
            self.bounds_max[1] - 0.02 * self.scene_extents[1],
            self.scene_center[2],
        ], dtype=float)
        self.plane_u, self.plane_v = self._compute_plane_axes(0.0, 0.0)
        self.plane_origin = self.plane_origin_base.copy()
        self.plane_size_u = max(1e-3, 1.5 * float(self.scene_extents[0]))
        self.plane_size_v = max(1e-3, 1.5 * float(self.scene_extents[2]))

        self.depth = max(0.05, 0.10 * float(max(self.scene_extents[1], 1e-3)))
        self.extrude_sign = -1.0

        # Face state -------------------------------------------------------
        self.face = FaceSelection()
        self.face_offset_u = 0.0
        self.face_offset_v = 0.0

        # Joint state ------------------------------------------------------
        self.current_cuboid: OrientedCuboid | None = None
        self.current_edge: Edge | None = None
        self.current_joint_type: str | None = None
        self.current_joint_limits: JointLimits | None = None
        self.last_pick_world: np.ndarray | None = None

        self.parent_mesh_stl: str | Path | None = None
        self.child_mesh_stl: str | Path | None = None

        self._staged_parent_mesh: trimesh.Trimesh | None = None
        self._staged_child_mesh: trimesh.Trimesh | None = None

        # Actors -----------------------------------------------------------
        self.mesh_actor = None
        self.plane_actor = None
        self.face_actor = None
        self.box_actor = None
        self.edge_actor = None

        self._build_ui()
        self._build_scene()

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    @property
    def plane_n(self) -> np.ndarray:
        return normalize(np.cross(self.plane_u, self.plane_v))

    def _compute_plane_axes(self, yaw_deg: float, pitch_deg: float):
        """Recompute plane_u / plane_v from scratch (no drift)."""
        base_u = np.array([1.0, 0.0, 0.0])
        base_v = np.array([0.0, 0.0, 1.0])

        R_yaw = rotation_matrix(np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg))
        u_yaw = normalize(R_yaw @ base_u)
        v_yaw = R_yaw @ base_v
        v_yaw = normalize(v_yaw - np.dot(v_yaw, u_yaw) * u_yaw)

        R_pitch = rotation_matrix(u_yaw, np.deg2rad(pitch_deg))
        u_final = normalize(R_pitch @ u_yaw)
        v_final = R_pitch @ v_yaw
        v_final = normalize(v_final - np.dot(v_final, u_final) * u_final)

        return u_final, v_final

    def _world_to_plane_uv(self, p_world: np.ndarray) -> np.ndarray:
        d = p_world - self.plane_origin
        return np.array([np.dot(d, self.plane_u), np.dot(d, self.plane_v)], dtype=float)

    def _plane_uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        return self.plane_origin + uv[0] * self.plane_u + uv[1] * self.plane_v

    def _effective_face(self):
        """Return (p0_uv, p1_uv) with face offset applied, or (None, None)."""
        if self.face.p0_uv is None or self.face.p1_uv is None:
            return None, None
        off = np.array([self.face_offset_u, self.face_offset_v], dtype=float)
        return self.face.p0_uv + off, self.face.p1_uv + off

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 3-D viewport
        self.plotter = pyvistaqt.QtInteractor(self)
        self.plotter.add_axes()
        self.plotter.show_grid()
        main_layout.addWidget(self.plotter, stretch=3)

        # Right control panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setSpacing(6)
        pl.setContentsMargins(8, 8, 8, 8)

        d = self.scene_diag

        # ---- Plane ----
        plane_grp = QGroupBox("Construction Plane")
        plane_lay = QVBoxLayout(plane_grp)

        self._w_plane_pos = SliderSpinBox(
            "Position along normal", -2 * d, 2 * d, 0.0, decimals=4
        )
        self._w_yaw = SliderSpinBox(
            "Yaw ° (about world Z)", -180.0, 180.0, 0.0, decimals=1
        )
        self._w_pitch = SliderSpinBox(
            "Pitch ° (about plane U)", -90.0, 90.0, 0.0, decimals=1
        )
        for w in (self._w_plane_pos, self._w_yaw, self._w_pitch):
            plane_lay.addWidget(w)

        view_row = QHBoxLayout()
        btn_vf = QPushButton("View Front")
        btn_vb = QPushButton("View Back")
        btn_vf.clicked.connect(lambda: self._view_plane_head_on(1.0))
        btn_vb.clicked.connect(lambda: self._view_plane_head_on(-1.0))
        view_row.addWidget(btn_vf)
        view_row.addWidget(btn_vb)
        plane_lay.addLayout(view_row)
        pl.addWidget(plane_grp)

        # ---- Cuboid ----
        cuboid_grp = QGroupBox("Cuboid")
        cuboid_lay = QVBoxLayout(cuboid_grp)

        self._w_depth = SliderSpinBox("Depth", 1e-3, 3 * d, self.depth, decimals=4)
        self._w_face_u = SliderSpinBox("Face offset U", -2 * d, 2 * d, 0.0, decimals=4)
        self._w_face_v = SliderSpinBox("Face offset V", -2 * d, 2 * d, 0.0, decimals=4)
        for w in (self._w_depth, self._w_face_u, self._w_face_v):
            cuboid_lay.addWidget(w)

        btn_flip = QPushButton("Flip Extrusion Direction")
        btn_flip.clicked.connect(self._flip_extrusion_direction)
        cuboid_lay.addWidget(btn_flip)
        pl.addWidget(cuboid_grp)

        # ---- Joint ----
        joint_grp = QGroupBox("Joint")
        joint_lay = QVBoxLayout(joint_grp)

        jbtn_row = QHBoxLayout()
        btn_hinge = QPushButton("Select Hinge (Revolute)")
        btn_slider_j = QPushButton("Select Slider (Prismatic)")
        btn_hinge.clicked.connect(self._choose_hinge)
        btn_slider_j.clicked.connect(self._choose_slider)
        jbtn_row.addWidget(btn_hinge)
        jbtn_row.addWidget(btn_slider_j)
        joint_lay.addLayout(jbtn_row)

        self._lbl_joint = QLabel("Joint: none  |  Edge: not selected")
        self._lbl_joint.setWordWrap(True)
        joint_lay.addWidget(self._lbl_joint)

        limits_grp = QGroupBox("Joint Limits")
        limits_lay = QVBoxLayout(limits_grp)
        self._lbl_limits_unit = QLabel("Select joint type above to set units.")
        self._lbl_limits_unit.setWordWrap(True)
        self._w_lower = SliderSpinBox("Lower limit", -360.0, 360.0, -90.0, decimals=3)
        self._w_upper = SliderSpinBox("Upper limit", -360.0, 360.0,  90.0, decimals=3)
        limits_lay.addWidget(self._lbl_limits_unit)
        limits_lay.addWidget(self._w_lower)
        limits_lay.addWidget(self._w_upper)
        joint_lay.addWidget(limits_grp)
        pl.addWidget(joint_grp)

        # ---- Actions ----
        actions_grp = QGroupBox("Actions")
        actions_lay = QVBoxLayout(actions_grp)
        btn_reset = QPushButton("Reset Face Selection")
        btn_door  = QPushButton("Door")
        btn_save  = QPushButton("Save Split Meshes")
        btn_urdf  = QPushButton("Create URDF")
        btn_print = QPushButton("Print Cuboid Info")
        btn_reset.clicked.connect(self._reset_face)
        btn_door.clicked.connect(self._apply_door)
        btn_save.clicked.connect(self._save_split_meshes)
        btn_urdf.clicked.connect(self._create_urdf_file)
        btn_print.clicked.connect(self._print_current_cuboid)
        for b in (btn_reset, btn_door, btn_save, btn_urdf, btn_print):
            actions_lay.addWidget(b)
        pl.addWidget(actions_grp)

        # Status label
        self._lbl_status = QLabel("Click twice on the blue plane to define a face.")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #555; font-style: italic; padding: 4px;")
        pl.addWidget(self._lbl_status)
        pl.addStretch()

        scroll.setWidget(panel)
        main_layout.addWidget(scroll, stretch=2)

        # Connect slider signals
        self._w_plane_pos.valueChanged.connect(self._on_plane_pos_changed)
        self._w_yaw.valueChanged.connect(self._on_rotation_changed)
        self._w_pitch.valueChanged.connect(self._on_rotation_changed)
        self._w_depth.valueChanged.connect(self._on_depth_changed)
        self._w_face_u.valueChanged.connect(self._on_face_offset_changed)
        self._w_face_v.valueChanged.connect(self._on_face_offset_changed)

    # -----------------------------------------------------------------------
    # Scene construction
    # -----------------------------------------------------------------------

    def _build_scene(self) -> None:
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_pv, color="lightgray", opacity=1.0,
            show_edges=False, name="object_mesh", pickable=False,
        )
        self.plotter.enable_surface_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_point=False,
            show_message=False,
        )
        self._update_plane()

    # -----------------------------------------------------------------------
    # Slider signal handlers
    # -----------------------------------------------------------------------

    def _on_plane_pos_changed(self, val: float) -> None:
        self.plane_offset = val
        self.plane_origin = self.plane_origin_base + self.plane_offset * self.plane_n
        self._update_plane()

    def _on_rotation_changed(self, _=None) -> None:
        self.yaw_deg = self._w_yaw.value()
        self.pitch_deg = self._w_pitch.value()
        self.plane_u, self.plane_v = self._compute_plane_axes(self.yaw_deg, self.pitch_deg)
        self.plane_origin = self.plane_origin_base + self.plane_offset * self.plane_n
        self._update_plane()

    def _on_depth_changed(self, val: float) -> None:
        self.depth = max(1e-4, val)
        had_joint = self.current_joint_type is not None
        self._update_cuboid_preview()
        self.plotter.render()
        if had_joint:
            self._set_status("[warn] Depth changed — joint/edge selection cleared. Reselect.")

    def _on_face_offset_changed(self, _=None) -> None:
        self.face_offset_u = self._w_face_u.value()
        self.face_offset_v = self._w_face_v.value()
        had_joint = self.current_joint_type is not None
        self._update_face_preview()
        self._update_cuboid_preview()
        self.plotter.render()
        if had_joint:
            self._set_status("[warn] Face moved — joint/edge selection cleared. Reselect.")

    # -----------------------------------------------------------------------
    # Plane / face rendering
    # -----------------------------------------------------------------------

    def _make_plane_mesh(self) -> pv.PolyData:
        hu, hv = self.plane_size_u, self.plane_size_v
        corners = np.array([
            self.plane_origin - hu*self.plane_u - hv*self.plane_v,
            self.plane_origin + hu*self.plane_u - hv*self.plane_v,
            self.plane_origin + hu*self.plane_u + hv*self.plane_v,
            self.plane_origin - hu*self.plane_u + hv*self.plane_v,
        ], dtype=float)
        return pv.PolyData(corners, np.hstack([[4, 0, 1, 2, 3]]))

    def _update_plane(self) -> None:
        if self.plane_actor is not None:
            self.plotter.remove_actor(self.plane_actor)
        self.plane_actor = self.plotter.add_mesh(
            self._make_plane_mesh(),
            color="deepskyblue", opacity=0.25, show_edges=True,
            name="construction_plane", pickable=True,
        )
        self._update_face_preview()
        self._update_cuboid_preview()
        self.plotter.render()

    def _update_face_preview(self) -> None:
        if self.face_actor is not None:
            self.plotter.remove_actor(self.face_actor)
            self.face_actor = None

        p0, p1 = self._effective_face()
        if p0 is None:
            return

        u0, v0 = p0
        u1, v1 = p1
        umin, umax = sorted([u0, u1])
        vmin, vmax = sorted([v0, v1])
        if (umax - umin) <= 1e-9 or (vmax - vmin) <= 1e-9:
            return

        corners_uv = [[umin, vmin], [umax, vmin], [umax, vmax], [umin, vmax]]
        corners_world = np.array([self._plane_uv_to_world(np.array(uv)) for uv in corners_uv])
        quad = pv.PolyData(corners_world, np.hstack([[4, 0, 1, 2, 3]]))
        self.face_actor = self.plotter.add_mesh(
            quad, color="orange", opacity=0.45, show_edges=True,
            line_width=3, name="face_preview",
        )

    def _update_cuboid_preview(self) -> None:
        for attr in ("box_actor", "edge_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)

        self.current_edge = None
        self.current_joint_type = None
        self.current_joint_limits = None
        self.current_cuboid = None
        self._staged_parent_mesh = None
        self._staged_child_mesh = None
        self._lbl_joint.setText("Joint: none  |  Edge: not selected")

        p0, p1 = self._effective_face()
        if p0 is None:
            return

        u0, v0 = p0
        u1, v1 = p1
        umin, umax = sorted([u0, u1])
        vmin, vmax = sorted([v0, v1])

        center_uv = np.array([0.5 * (umin + umax), 0.5 * (vmin + vmax)])
        face_center = self._plane_uv_to_world(center_uv)
        half_u = 0.5 * (umax - umin)
        half_v = 0.5 * (vmax - vmin)
        half_n = 0.5 * self.depth

        if half_u <= 1e-9 or half_v <= 1e-9 or half_n <= 1e-9:
            return

        n = self.plane_n
        self.current_cuboid = OrientedCuboid(
            center=face_center + self.extrude_sign * half_n * n,
            rotation=np.column_stack([self.plane_u, self.plane_v, n]),
            extents=np.array([half_u, half_v, half_n], dtype=float),
        )

        corners_local = np.array([
            [-half_u, -half_v, -half_n], [ half_u, -half_v, -half_n],
            [ half_u,  half_v, -half_n], [-half_u,  half_v, -half_n],
            [-half_u, -half_v,  half_n], [ half_u, -half_v,  half_n],
            [ half_u,  half_v,  half_n], [-half_u,  half_v,  half_n],
        ], dtype=float)
        corners_world = self.current_cuboid.local_to_world(corners_local)

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
        self.box_actor = self.plotter.add_mesh(wire, color="red", line_width=3, name="cuboid_preview")

    # -----------------------------------------------------------------------
    # Pick callback
    # -----------------------------------------------------------------------

    def _on_pick(self, point: np.ndarray) -> None:
        try:
            p_world = np.array(point, dtype=float)
            self.last_pick_world = p_world
            uv = self._world_to_plane_uv(p_world)

            if self.face.p0_uv is None:
                self.face.p0_uv = uv
                self._set_status("p0 set — click to set p1.")
            elif self.face.p1_uv is None:
                self.face.p1_uv = uv
                self._set_status("Face defined. Select joint type or reset.")
            else:
                self._set_status("Face already set. Use offset sliders to fine-tune, or reset.")
                return

            self._update_face_preview()
            self._update_cuboid_preview()
            self.plotter.render()
        except Exception as exc:
            self._set_status(f"[error] pick: {exc}")

    # -----------------------------------------------------------------------
    # Button actions
    # -----------------------------------------------------------------------

    def _flip_extrusion_direction(self) -> None:
        self.extrude_sign *= -1.0
        had_joint = self.current_joint_type is not None
        self._update_cuboid_preview()
        self.plotter.render()
        msg = f"Extrusion: {'+normal' if self.extrude_sign > 0 else '-normal'}"
        if had_joint:
            msg += " — joint/edge selection cleared. Reselect."
        self._set_status(msg)

    def _view_plane_head_on(self, side: float) -> None:
        try:
            n = self.plane_n
            up = normalize(self.plane_v)
            pos = np.array(self.plotter.camera.position, dtype=float)
            dist = np.linalg.norm(pos - self.plane_origin)
            if dist < 1e-6:
                dist = max(1.0, 1.2 * self.scene_diag)
            self.plotter.camera.position = tuple(self.plane_origin + side * dist * n)
            self.plotter.camera.focal_point = tuple(self.plane_origin)
            self.plotter.camera.up = tuple(up)
            self.plotter.render()
        except Exception as exc:
            self._set_status(f"[error] view head-on: {exc}")

    def _reset_face(self) -> None:
        self.face = FaceSelection()
        self.face_offset_u = 0.0
        self.face_offset_v = 0.0
        self._w_face_u.set_value(0.0)
        self._w_face_v.set_value(0.0)
        self.current_cuboid = None
        self.current_edge = None
        self.current_joint_type = None
        self.current_joint_limits = None
        self.last_pick_world = None
        self._staged_parent_mesh = None
        self._staged_child_mesh = None
        self._lbl_joint.setText("Joint: none  |  Edge: not selected")

        for attr in ("face_actor", "box_actor", "edge_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)

        self.plotter.render()
        self._set_status("Face reset. Click twice on the blue plane to define a new face.")

    def _choose_edge_joint(self, joint_type: str) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.last_pick_world is None:
            self._set_status("[warn] Click near a cuboid edge first, then select joint type.")
            return

        # Find nearest edge to last click
        half_u, half_v, half_n = self.current_cuboid.extents
        corners_local = np.array([
            [-half_u, -half_v, -half_n], [ half_u, -half_v, -half_n],
            [ half_u,  half_v, -half_n], [-half_u,  half_v, -half_n],
            [-half_u, -half_v,  half_n], [ half_u, -half_v,  half_n],
            [ half_u,  half_v,  half_n], [-half_u,  half_v,  half_n],
        ], dtype=float)
        corners_world = self.current_cuboid.local_to_world(corners_local)
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        pick = self.last_pick_world
        best_ep, best_dist = edge_pairs[0], float("inf")
        for ep in edge_pairs:
            p0w, p1w = corners_world[ep[0]], corners_world[ep[1]]
            seg = p1w - p0w
            sq = float(np.dot(seg, seg))
            t = float(np.clip(np.dot(pick - p0w, seg) / sq, 0, 1)) if sq > 1e-12 else 0.0
            dist = float(np.linalg.norm(pick - (p0w + t * seg)))
            if dist < best_dist:
                best_dist = dist
                best_ep = ep

        # Read limits from UI
        lower_ui = self._w_lower.value()
        upper_ui = self._w_upper.value()

        if lower_ui >= upper_ui:
            self._set_status("[error] Lower limit must be strictly less than upper limit.")
            return

        if joint_type == "revolute":
            lower = np.deg2rad(lower_ui)
            upper = np.deg2rad(upper_ui)
            unit = "rad"
            limits_str = f"[{lower_ui:.1f}°, {upper_ui:.1f}°]"
        else:
            lower = lower_ui
            upper = upper_ui
            unit = "m"
            limits_str = f"[{lower:.4f} m, {upper:.4f} m]"

        edge = Edge(p0_world=corners_world[best_ep[0]], p1_world=corners_world[best_ep[1]])
        self.current_edge = edge
        self.current_joint_type = joint_type
        self.current_joint_limits = JointLimits(lower=lower, upper=upper, unit=unit)

        self._lbl_joint.setText(f"Joint: {joint_type}  |  Limits: {limits_str}")

        if self.edge_actor is not None:
            self.plotter.remove_actor(self.edge_actor)
        edge_line = pv.Line(edge.p0_world, edge.p1_world)
        self.edge_actor = self.plotter.add_mesh(
            edge_line, color="lime", line_width=8, name="edge_selection"
        )
        self.plotter.render()
        self._set_status(f"[{joint_type}] edge selected. Limits: {limits_str}")

    def _choose_hinge(self) -> None:
        self._w_lower.set_range(-360.0, 360.0)
        self._w_upper.set_range(-360.0, 360.0)
        self._lbl_limits_unit.setText("Revolute: values in degrees")
        self._choose_edge_joint("revolute")

    def _choose_slider(self) -> None:
        d = self.scene_diag
        self._w_lower.set_range(-2 * d, 2 * d)
        self._w_upper.set_range(-2 * d, 2 * d)
        self._lbl_limits_unit.setText("Prismatic: values in meters")
        self._choose_edge_joint("prismatic")

    def _print_current_cuboid(self) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        print("\nCurrent cuboid:")
        print(f"center = np.array({self.current_cuboid.center.tolist()})")
        print(f"rotation = np.array({self.current_cuboid.rotation.tolist()})")
        print(f"extents = np.array({self.current_cuboid.extents.tolist()})\n")
        self._set_status("Cuboid info printed to console.")

    def _apply_door(self) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.current_edge is None:
            self._set_status("[warn] Select joint type first.")
            return
        try:
            result = split_mesh_by_cuboid_clip(self.mesh_tm, self.current_cuboid)
            hinge = np.asarray(self.current_edge.p0_world, dtype=float)
            door_mesh, _ = cut_cuboid_with_surface(result.inside_mesh, self.current_cuboid)
            door_mesh.vertices = door_mesh.vertices - hinge
            parent_mesh = result.outside_mesh.copy()
            self._staged_parent_mesh = parent_mesh
            self._staged_child_mesh = door_mesh
            self._set_status(
                f"Door prepared — inside: {len(door_mesh.faces)} faces, "
                f"outside: {len(parent_mesh.faces)} faces. Press Save to write."
            )
        except Exception as exc:
            self._set_status(f"[error] door: {exc}")

    def _save_split_meshes(self) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.current_edge is None:
            self._set_status("[warn] Select joint type first.")
            return
        try:
            out_dir = Path("data/output/split_test")
            out_dir.mkdir(parents=True, exist_ok=True)
            inside_path = out_dir / "selection_clip.stl"
            outside_path = out_dir / "rest_clip.stl"

            hinge = np.asarray(self.current_edge.p0_world, dtype=float)

            if self._staged_parent_mesh is not None and self._staged_child_mesh is not None:
                parent_mesh = self._staged_parent_mesh
                child_mesh = self._staged_child_mesh
            else:
                result = split_mesh_by_cuboid_clip(self.mesh_tm, self.current_cuboid)
                parent_mesh = result.outside_mesh.copy()
                child_mesh = result.inside_mesh.copy()
                child_mesh.vertices = child_mesh.vertices - hinge

            parent_mesh.export(outside_path)
            child_mesh.export(inside_path)

            self.parent_mesh_stl = outside_path
            self.child_mesh_stl = inside_path
            self._set_status(
                f"Saved — inside: {len(child_mesh.faces)} faces, "
                f"outside: {len(parent_mesh.faces)} faces."
            )
        except Exception as exc:
            self._set_status(f"[error] save: {exc}")

    def _create_urdf_file(self) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet."); return
        if self.current_edge is None or self.current_joint_type is None:
            self._set_status("[warn] Select joint type first."); return
        if self.current_joint_limits is None:
            self._set_status("[warn] Joint limits required."); return
        if self.parent_mesh_stl is None or self.child_mesh_stl is None:
            self._set_status("[warn] Save split meshes first."); return

        export_dir = Path("data/output/urdf_test")
        export_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = export_dir / "name.urdf"
        try:
            export_to_urdf(
                urdf_path=urdf_path,
                parent_mesh_stl=self.parent_mesh_stl,
                child_mesh_stl=self.child_mesh_stl,
                cuboid=self.current_cuboid,
                edge_of_interest=self.current_edge,
                joint_type=self.current_joint_type,
                joint_limits=self.current_joint_limits,
            )
            self._set_status(f"URDF saved to {urdf_path}")
        except Exception as exc:
            self._set_status(f"[error] URDF export: {exc}")

    # -----------------------------------------------------------------------
    # Status helper
    # -----------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._lbl_status.setText(msg)
        print(f"[status] {msg}")

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def run(self) -> None:
        self.resize(1400, 800)
        self.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    window = CuboidSelectorApp("data/input/stove.stl")
    window.run()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
