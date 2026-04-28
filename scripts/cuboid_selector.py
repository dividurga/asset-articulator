from __future__ import annotations

import sys
import itertools
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pyvista as pv
import pyvistaqt
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QSlider, QDoubleSpinBox, QPushButton, QScrollArea,
    QCheckBox, QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, pyqtSignal

from asset_articulator.assets.joints import JointLimits
from asset_articulator.geometry.clip import split_mesh_by_cuboid_clip, split_mesh_by_cylinder_clip
from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.cylinder import OrientedCylinder
from asset_articulator.geometry.edge import Edge
from asset_articulator.geometry.door import cut_cuboid_with_surface, cut_cylinder_with_surface
from asset_articulator.io.urdf_export import ArticulationSpec, export_to_urdf


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


@dataclass
class ArticulationEntry:
    """One completed door/slider selection ready for URDF export."""
    edge: Edge
    joint_type: str
    joint_limits: JointLimits
    door_mesh: trimesh.Trimesh   # in hinge-local frame (shifted by -hinge_origin)
    hinge_origin: np.ndarray     # world coords of joint origin
    cuboid: OrientedCuboid | None = None
    cylinder: OrientedCylinder | None = None
    box_actor: object = None     # pyvista actor for this door's wireframe


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
        self.roll_deg = 0.0
        self.plane_offset = 0.0
        self.plane_origin_base = np.array([
            self.scene_center[0],
            self.bounds_max[1] - 0.02 * self.scene_extents[1],
            self.scene_center[2],
        ], dtype=float)
        self.plane_u, self.plane_v = self._compute_plane_axes(0.0, 0.0, 0.0)
        self.plane_origin = self.plane_origin_base.copy()
        self.plane_size_u = max(1e-3, 1.5 * float(self.scene_extents[0]))
        self.plane_size_v = max(1e-3, 1.5 * float(self.scene_extents[2]))

        self.depth = max(0.05, 0.10 * float(max(self.scene_extents[1], 1e-3)))
        self.extrude_sign = -1.0

        # Face state -------------------------------------------------------
        self.face = FaceSelection()
        self.face_offset_u = 0.0
        self.face_offset_v = 0.0
        self._face_center_uv: np.ndarray | None = None

        # Selection mode ---------------------------------------------------
        self._selection_mode: str = "cuboid"  # "cuboid", "cylinder", or "cabinet"
        self._cyl_center_uv: np.ndarray | None = None
        self._cyl_radius: float = 0.0

        # Cabinet mode state -----------------------------------------------
        self._cab_split_u: float | None = None   # split coord when axis == "u" (vertical split)
        self._cab_split_v: float | None = None   # split coord when axis == "v" (horizontal split)
        self._cab_split_axis: str = "u"          # "u" = vertical split, "v" = horizontal split
        self._cab_snap_pt_uv: np.ndarray | None = None  # edge snap point in plane UV
        self._cab_left_cuboid: OrientedCuboid | None = None
        self._cab_right_cuboid: OrientedCuboid | None = None
        self._cab_left_edge: Edge | None = None
        self._cab_right_edge: Edge | None = None
        self._cab_left_limits: JointLimits | None = None
        self._cab_right_limits: JointLimits | None = None
        self._cab_state: str = "face"   # "face" | "split" | "hinge"
        self._cab_hinge_target: str | None = None       # "left" | "right"
        self._cab_split_actor = None
        self._cab_snap_pt_actor = None
        self._cab_left_box_actor = None
        self._cab_right_box_actor = None
        self._cab_left_edge_actor = None
        self._cab_right_edge_actor = None

        # Joint state ------------------------------------------------------
        self._joint_armed: str | None = None  # "revolute" | "prismatic" | None
        self.current_cuboid: OrientedCuboid | None = None
        self.current_cylinder: OrientedCylinder | None = None
        self.current_edge: Edge | None = None
        self.current_joint_type: str | None = None
        self.current_joint_limits: JointLimits | None = None
        self.last_pick_world: np.ndarray | None = None

        # Multi-door articulation list
        self._articulations: list[ArticulationEntry] = []

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

    def _compute_plane_axes(self, yaw_deg: float, pitch_deg: float, roll_deg: float):
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

        if abs(roll_deg) > 1e-12:
            n_final = normalize(np.cross(u_final, v_final))
            R_roll = rotation_matrix(n_final, np.deg2rad(roll_deg))
            u_final = normalize(R_roll @ u_final)
            v_final = normalize(R_roll @ v_final)

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

        # ---- Selection Mode ----
        mode_grp = QGroupBox("Selection Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self._rb_cuboid = QRadioButton("Cuboid")
        self._rb_cylinder = QRadioButton("Cylinder")
        self._rb_cabinet = QRadioButton("Cabinet")
        self._rb_cuboid.setChecked(True)
        self._btn_grp_mode = QButtonGroup()
        self._btn_grp_mode.addButton(self._rb_cuboid)
        self._btn_grp_mode.addButton(self._rb_cylinder)
        self._btn_grp_mode.addButton(self._rb_cabinet)
        mode_lay.addWidget(self._rb_cuboid)
        mode_lay.addWidget(self._rb_cylinder)
        mode_lay.addWidget(self._rb_cabinet)
        pl.addWidget(mode_grp)

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

        self._w_roll = SliderSpinBox(
            "Roll ° (about plane normal)", -180.0, 180.0, 0.0, decimals=1
        )
        plane_lay.addWidget(self._w_roll)

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
        self._w_size_u = SliderSpinBox("Width (U extent)", 1e-4, 3 * d, 0.1 * d, decimals=4)
        self._w_size_v = SliderSpinBox("Height (V extent)", 1e-4, 3 * d, 0.1 * d, decimals=4)
        self._w_face_u = SliderSpinBox("Face offset U", -2 * d, 2 * d, 0.0, decimals=4)
        self._w_face_v = SliderSpinBox("Face offset V", -2 * d, 2 * d, 0.0, decimals=4)
        for w in (self._w_depth, self._w_size_u, self._w_size_v, self._w_face_u, self._w_face_v):
            cuboid_lay.addWidget(w)

        btn_flip = QPushButton("Flip Extrusion Direction")
        btn_flip.clicked.connect(self._flip_extrusion_direction)
        cuboid_lay.addWidget(btn_flip)
        pl.addWidget(cuboid_grp)

        # ---- Joint ----
        joint_grp = QGroupBox("Joint")
        joint_lay = QVBoxLayout(joint_grp)

        self._limits_grp = QGroupBox("Joint Limits")
        limits_grp = self._limits_grp
        limits_lay = QVBoxLayout(limits_grp)
        self._lbl_limits_unit = QLabel("Set limits, then click a joint button below.")
        self._lbl_limits_unit.setWordWrap(True)
        self._w_lower = SliderSpinBox("Lower limit", -360.0, 360.0, -90.0, decimals=3)
        self._w_upper = SliderSpinBox("Upper limit", -360.0, 360.0,  90.0, decimals=3)
        limits_lay.addWidget(self._lbl_limits_unit)
        limits_lay.addWidget(self._w_lower)
        limits_lay.addWidget(self._w_upper)
        joint_lay.addWidget(limits_grp)

        jbtn_row = QHBoxLayout()
        self._btn_hinge = QPushButton("Select Hinge (Revolute)")
        self._btn_slider_j = QPushButton("Select Slider (Prismatic)")
        self._btn_hinge.clicked.connect(self._choose_hinge)
        self._btn_slider_j.clicked.connect(self._choose_slider)
        jbtn_row.addWidget(self._btn_hinge)
        jbtn_row.addWidget(self._btn_slider_j)
        joint_lay.addLayout(jbtn_row)

        self._btn_axis = QPushButton("Select Axis (Revolute)")
        self._btn_axis.clicked.connect(self._choose_cylinder_axis)
        self._btn_axis.setEnabled(False)
        joint_lay.addWidget(self._btn_axis)

        self._lbl_joint = QLabel("Joint: none  |  Edge: not selected")
        self._lbl_joint.setWordWrap(True)
        joint_lay.addWidget(self._lbl_joint)

        pl.addWidget(joint_grp)

        # ---- Actions ----
        actions_grp = QGroupBox("Actions")
        actions_lay = QVBoxLayout(actions_grp)
        btn_reset     = QPushButton("Reset Face Selection")
        self._btn_preview_door = QPushButton("Preview Door")
        self._btn_add_door     = QPushButton("Add Door")
        self._btn_preview_cyl  = QPushButton("Preview Cylinder")
        self._btn_add_cyl      = QPushButton("Add Cylinder")
        btn_clear     = QPushButton("Clear All Selections")
        btn_urdf      = QPushButton("Export URDF")
        btn_print     = QPushButton("Print Cuboid Info")
        btn_reset.clicked.connect(self._reset_face)
        self._btn_preview_door.clicked.connect(self._apply_door)
        self._btn_add_door.clicked.connect(self._add_door)
        self._btn_preview_cyl.clicked.connect(self._apply_cylinder)
        self._btn_add_cyl.clicked.connect(self._add_cylinder)
        self._btn_preview_cyl.setEnabled(False)
        self._btn_add_cyl.setEnabled(False)
        btn_clear.clicked.connect(self._clear_doors)
        btn_urdf.clicked.connect(self._create_urdf_file)
        btn_print.clicked.connect(self._print_current_cuboid)
        self._chk_slice_only = QCheckBox("Slice only (export single combined mesh)")
        self._lbl_doors = QLabel("Doors queued: 0")
        for b in (btn_reset, self._btn_preview_door, self._btn_add_door,
                  self._btn_preview_cyl, self._btn_add_cyl,
                  btn_clear, btn_urdf, btn_print):
            actions_lay.addWidget(b)
        actions_lay.addWidget(self._chk_slice_only)
        actions_lay.addWidget(self._lbl_doors)
        pl.addWidget(actions_grp)

        # ---- Cabinet Split ----
        self._cabinet_grp = QGroupBox("Cabinet Split")
        cabinet_lay = QVBoxLayout(self._cabinet_grp)

        self._lbl_cab_step = QLabel("Step 1: Click twice to define face.")
        self._lbl_cab_step.setWordWrap(True)
        cabinet_lay.addWidget(self._lbl_cab_step)

        self._btn_cab_resplit = QPushButton("Re-pick Split Line")
        self._btn_cab_resplit.clicked.connect(self._cab_enter_split_mode)
        self._btn_cab_resplit.setEnabled(False)
        cabinet_lay.addWidget(self._btn_cab_resplit)

        hinge_row = QHBoxLayout()
        self._btn_cab_left = QPushButton("Select Left Hinge")
        self._btn_cab_right = QPushButton("Select Right Hinge")
        self._btn_cab_left.clicked.connect(lambda: self._cab_arm_hinge("left"))
        self._btn_cab_right.clicked.connect(lambda: self._cab_arm_hinge("right"))
        self._btn_cab_left.setEnabled(False)
        self._btn_cab_right.setEnabled(False)
        hinge_row.addWidget(self._btn_cab_left)
        hinge_row.addWidget(self._btn_cab_right)
        cabinet_lay.addLayout(hinge_row)

        self._lbl_cab_edges = QLabel("Left: none  |  Right: none")
        self._lbl_cab_edges.setWordWrap(True)
        cabinet_lay.addWidget(self._lbl_cab_edges)

        self._btn_add_cabinet = QPushButton("Add Cabinet")
        self._btn_add_cabinet.clicked.connect(self._add_cabinet)
        self._btn_add_cabinet.setEnabled(False)
        cabinet_lay.addWidget(self._btn_add_cabinet)

        pl.addWidget(self._cabinet_grp)
        self._cabinet_grp.setVisible(False)

        # Status label
        self._lbl_status = QLabel("Click twice on the blue plane to define a face.")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #555; font-style: italic; padding: 4px;")
        pl.addWidget(self._lbl_status)
        pl.addStretch()

        scroll.setWidget(panel)
        main_layout.addWidget(scroll, stretch=2)

        # Connect mode radio
        self._rb_cuboid.toggled.connect(self._on_mode_changed)
        self._rb_cabinet.toggled.connect(self._on_mode_changed)

        # Connect slider signals
        self._w_plane_pos.valueChanged.connect(self._on_plane_pos_changed)
        self._w_yaw.valueChanged.connect(self._on_rotation_changed)
        self._w_pitch.valueChanged.connect(self._on_rotation_changed)
        self._w_roll.valueChanged.connect(self._on_rotation_changed)
        self._w_depth.valueChanged.connect(self._on_depth_changed)
        self._w_size_u.valueChanged.connect(self._on_face_size_changed)
        self._w_size_v.valueChanged.connect(self._on_face_size_changed)
        self._w_face_u.valueChanged.connect(self._on_face_offset_changed)
        self._w_face_v.valueChanged.connect(self._on_face_offset_changed)

    # -----------------------------------------------------------------------
    # Scene construction
    # -----------------------------------------------------------------------

    def _build_scene(self) -> None:
        self._xray_on = False
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_pv, color="lightgray", opacity=1.0,
            show_edges=False, name="object_mesh", pickable=False,
        )
        self._wire_actor = self.plotter.add_mesh(
            self.mesh_pv, color="#555555", style="wireframe",
            line_width=0.5, name="object_wire", pickable=False,
        )
        self._wire_actor.SetVisibility(False)

        def _toggle_xray(state: bool) -> None:
            self._xray_on = state
            self.mesh_actor.GetProperty().SetOpacity(0.15 if state else 1.0)
            self._wire_actor.SetVisibility(state)
            self.plotter.render()

        self.plotter.add_checkbox_button_widget(
            _toggle_xray,
            value=False,
            position=(10, self.plotter.window_size[1] - 50),
            size=30,
            border_size=2,
            color_on="#9b59d0",
            color_off="#aaaaaa",
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
        self.roll_deg = self._w_roll.value()
        self.plane_u, self.plane_v = self._compute_plane_axes(self.yaw_deg, self.pitch_deg, self.roll_deg)
        self.plane_origin = self.plane_origin_base + self.plane_offset * self.plane_n
        self._update_plane()

    def _on_depth_changed(self, val: float) -> None:
        self.depth = max(1e-4, val)
        had_joint = self.current_joint_type is not None
        if self._selection_mode == "cylinder":
            self._update_cylinder_preview()
        elif self._selection_mode == "cabinet":
            self._update_cabinet_preview()
        else:
            self._update_cuboid_preview()
        self.plotter.render()
        if had_joint:
            self._set_status("[warn] Depth changed — joint/edge selection cleared. Reselect.")

    def _on_face_offset_changed(self, _=None) -> None:
        self.face_offset_u = self._w_face_u.value()
        self.face_offset_v = self._w_face_v.value()
        had_joint = self.current_joint_type is not None
        if self._selection_mode == "cylinder":
            self._update_cylinder_preview()
        elif self._selection_mode == "cabinet":
            self._update_face_preview()
            self._update_cabinet_preview()
        else:
            self._update_face_preview()
            self._update_cuboid_preview()
        self.plotter.render()
        if had_joint:
            self._set_status("[warn] Face moved — joint/edge selection cleared. Reselect.")

    def _on_face_size_changed(self, _=None) -> None:
        if self._selection_mode == "cylinder":
            return
        # Use face center if available; fall back to first click point as center
        center = self._face_center_uv
        if center is None:
            if self.face.p0_uv is not None:
                center = self.face.p0_uv.copy()
                self._face_center_uv = center
            else:
                return
        hu = self._w_size_u.value() / 2.0
        hv = self._w_size_v.value() / 2.0
        self.face.p0_uv = center + np.array([-hu, -hv])
        self.face.p1_uv = center + np.array([hu, hv])
        had_joint = self.current_joint_type is not None
        self._update_face_preview()
        if self._selection_mode == "cabinet":
            self._update_cabinet_preview()
        else:
            self._update_cuboid_preview()
        self.plotter.render()
        if had_joint:
            self._set_status("[warn] Face resized — joint/edge selection cleared. Reselect.")

    # -----------------------------------------------------------------------
    # Plane / face rendering
    # -----------------------------------------------------------------------

    def _make_plane_mesh(self) -> pv.PolyData:
        # Project all 8 AABB corners onto current plane axes so the visual
        # always covers the full mesh regardless of plane orientation.
        bbox_corners = np.array(list(itertools.product(
            [self.bounds_min[0], self.bounds_max[0]],
            [self.bounds_min[1], self.bounds_max[1]],
            [self.bounds_min[2], self.bounds_max[2]],
        )))
        proj_u = (bbox_corners - self.plane_origin) @ self.plane_u
        proj_v = (bbox_corners - self.plane_origin) @ self.plane_v
        margin = 0.05 * self.scene_diag
        hu = max(abs(proj_u).max() + margin, 1e-3)
        hv = max(abs(proj_v).max() + margin, 1e-3)
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
            name="construction_plane", pickable=True, reset_camera=False,
        )
        if self._selection_mode == "cylinder":
            self._update_cylinder_preview()
        elif self._selection_mode == "cabinet":
            self._update_face_preview()
            self._update_cabinet_preview()
        else:
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
            line_width=3, name="face_preview", reset_camera=False,
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
        self._staged_child_mesh = None
        self._joint_armed = None
        self._btn_hinge.setText("Select Hinge (Revolute)")
        self._btn_slider_j.setText("Select Slider (Prismatic)")
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
        self.box_actor = self.plotter.add_mesh(wire, color="red", line_width=3, name="cuboid_preview", reset_camera=False)

    # -----------------------------------------------------------------------
    # Pick callback
    # -----------------------------------------------------------------------

    def _on_pick(self, point: np.ndarray) -> None:
        try:
            p_world = np.array(point, dtype=float)
            self.last_pick_world = p_world
            uv = self._world_to_plane_uv(p_world)

            if self._selection_mode == "cylinder":
                if self._cyl_center_uv is None:
                    self._cyl_center_uv = uv
                    self._set_status("Cylinder center set — click perimeter point.")
                elif self._cyl_radius <= 0:
                    r = float(np.linalg.norm(uv - self._cyl_center_uv))
                    if r < 1e-9:
                        self._set_status("[warn] Perimeter point too close to center.")
                        return
                    self._cyl_radius = r
                    self._set_status("Cylinder defined. Click 'Select Axis (Continuous)' or reset.")
                else:
                    self._set_status("Cylinder already set. Use offset sliders or reset.")
                    return
                self._update_cylinder_preview()
                self.plotter.render()
                return

            if self._selection_mode == "cabinet":
                self._on_cabinet_pick(uv, p_world)
                return

            if self.face.p0_uv is None:
                self.face.p0_uv = uv
                self._set_status("p0 set — click to set p1.")
            elif self.face.p1_uv is None:
                self.face.p1_uv = uv
                self._face_center_uv = 0.5 * (self.face.p0_uv + uv)
                size_u = abs(uv[0] - self.face.p0_uv[0])
                size_v = abs(uv[1] - self.face.p0_uv[1])
                self._w_size_u.set_value(size_u)
                self._w_size_v.set_value(size_v)
                self._set_status("Face defined. Click a joint button then click near an edge.")
            else:
                if self._joint_armed is not None and self.current_cuboid is not None:
                    self._choose_edge_joint(self._joint_armed)
                    self._joint_armed = None
                else:
                    self._set_status("Face already set. Use offset sliders to fine-tune, or reset.")
                return

            self._update_face_preview()
            self._update_cuboid_preview()
            self.plotter.render()
        except Exception as exc:
            self._set_status(f"[error] pick: {exc}")

    def _on_cabinet_pick(self, uv: np.ndarray, p_world: np.ndarray) -> None:
        if self._cab_state == "face":
            if self.face.p0_uv is None:
                self.face.p0_uv = uv
                self._lbl_cab_step.setText("Step 1: Click p1 (second corner).")
                self._set_status("Cabinet: p0 set — click to set p1.")
            elif self.face.p1_uv is None:
                self.face.p1_uv = uv
                self._face_center_uv = 0.5 * (self.face.p0_uv + uv)
                self._w_size_u.set_value(abs(uv[0] - self.face.p0_uv[0]))
                self._w_size_v.set_value(abs(uv[1] - self.face.p0_uv[1]))
                self._cab_state = "split"
                self._btn_cab_resplit.setEnabled(True)
                self._lbl_cab_step.setText("Step 2: Click on a face edge to set split line.")
                self._set_status("Cabinet: face defined — click on top/bottom edge for vertical split, left/right for horizontal.")
            else:
                self._set_status("Cabinet: face already set. Use sliders or Reset Face.")
                return
            self._update_face_preview()
            self._update_cabinet_preview()
            self.plotter.render()

        elif self._cab_state == "split":
            p0, p1 = self._effective_face()
            if p0 is None:
                return
            umin = min(p0[0], p1[0])
            umax = max(p0[0], p1[0])
            vmin = min(p0[1], p1[1])
            vmax = max(p0[1], p1[1])
            u, v = float(uv[0]), float(uv[1])

            # Snap click to the nearest face edge, then project onto it.
            # Nearest edge determines whether the split is vertical (top/bottom edge clicked)
            # or horizontal (left/right edge clicked).
            d_left   = abs(u - umin)
            d_right  = abs(u - umax)
            d_bottom = abs(v - vmin)
            d_top    = abs(v - vmax)
            nearest  = min(d_left, d_right, d_bottom, d_top)

            if nearest in (d_bottom, d_top):
                # Click was on top or bottom edge → vertical split line drops across face
                self._cab_split_axis = "u"
                self._cab_split_u = float(np.clip(u, umin + 1e-6, umax - 1e-6))
                self._cab_split_v = None
                snap_v = vmin if nearest == d_bottom else vmax
                self._cab_snap_pt_uv = np.array([self._cab_split_u, snap_v])
                self._btn_cab_left.setText("Select Left Hinge")
                self._btn_cab_right.setText("Select Right Hinge")
            else:
                # Click was on left or right edge → horizontal split line runs across face
                self._cab_split_axis = "v"
                self._cab_split_v = float(np.clip(v, vmin + 1e-6, vmax - 1e-6))
                self._cab_split_u = None
                snap_u = umin if nearest == d_left else umax
                self._cab_snap_pt_uv = np.array([snap_u, self._cab_split_v])
                self._btn_cab_left.setText("Select Bottom Hinge")
                self._btn_cab_right.setText("Select Top Hinge")

            self._cab_state = "hinge"
            self._btn_cab_left.setEnabled(True)
            self._btn_cab_right.setEnabled(True)
            self._update_cabinet_preview()
            self.plotter.render()
            axis_str = "vertical" if self._cab_split_axis == "u" else "horizontal"
            self._lbl_cab_step.setText("Step 3: Select the two hinge edges, then 'Add Cabinet'.")
            self._set_status(f"Cabinet: {axis_str} split set. Select hinges (orange / cyan).")

        elif self._cab_state == "hinge" and self._cab_hinge_target is not None:
            self._pick_cabinet_edge(self._cab_hinge_target)
            self._cab_hinge_target = None

    # -----------------------------------------------------------------------
    # Button actions
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Cuboid wire helper
    # -----------------------------------------------------------------------

    def _draw_cuboid_wire(self, cuboid: OrientedCuboid, color: str, name: str):
        half_u, half_v, half_n = cuboid.extents
        corners_local = np.array([
            [-half_u, -half_v, -half_n], [ half_u, -half_v, -half_n],
            [ half_u,  half_v, -half_n], [-half_u,  half_v, -half_n],
            [-half_u, -half_v,  half_n], [ half_u, -half_v,  half_n],
            [ half_u,  half_v,  half_n], [-half_u,  half_v,  half_n],
        ], dtype=float)
        corners_world = cuboid.local_to_world(corners_local)
        edge_pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        lines: list[int] = []
        for a, b in edge_pairs:
            lines.extend([2, a, b])
        wire = pv.PolyData()
        wire.points = corners_world
        wire.lines = np.array(lines, dtype=np.int32)
        return self.plotter.add_mesh(wire, color=color, line_width=3, name=name, reset_camera=False)

    # -----------------------------------------------------------------------
    # Cabinet helpers
    # -----------------------------------------------------------------------

    def _update_cabinet_preview(self) -> None:
        for attr in ("_cab_split_actor", "_cab_snap_pt_actor", "_cab_left_box_actor",
                     "_cab_right_box_actor", "box_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)

        self._cab_left_cuboid = None
        self._cab_right_cuboid = None

        p0, p1 = self._effective_face()
        if p0 is None:
            return

        umin = min(p0[0], p1[0])
        umax = max(p0[0], p1[0])
        vmin = min(p0[1], p1[1])
        vmax = max(p0[1], p1[1])

        if (umax - umin) <= 1e-9 or (vmax - vmin) <= 1e-9:
            return

        n = self.plane_n
        half_n = 0.5 * self.depth
        u_center = 0.5 * (umin + umax)
        v_center = 0.5 * (vmin + vmax)

        has_split = (self._cab_split_axis == "u" and self._cab_split_u is not None) or \
                    (self._cab_split_axis == "v" and self._cab_split_v is not None)

        if not has_split:
            # Unsplit outline while waiting for edge click
            outline = OrientedCuboid(
                center=self._plane_uv_to_world(np.array([u_center, v_center])) + self.extrude_sign * half_n * n,
                rotation=np.column_stack([self.plane_u, self.plane_v, n]),
                extents=np.array([0.5 * (umax - umin), 0.5 * (vmax - vmin), half_n], dtype=float),
            )
            self.box_actor = self._draw_cuboid_wire(outline, "red", "cab_outline")
            return

        # Draw snap-point marker on the clicked edge
        if self._cab_snap_pt_uv is not None:
            snap_world = self._plane_uv_to_world(self._cab_snap_pt_uv)
            self._cab_snap_pt_actor = self.plotter.add_mesh(
                pv.Sphere(radius=self.scene_diag * 0.006, center=snap_world),
                color="white", name="cab_snap_pt", reset_camera=False,
            )

        if self._cab_split_axis == "u":
            # Vertical split line: drops from bottom edge to top edge at split_u
            split_u = float(np.clip(self._cab_split_u, umin + 1e-6, umax - 1e-6))
            line_a = self._plane_uv_to_world(np.array([split_u, vmin]))
            line_b = self._plane_uv_to_world(np.array([split_u, vmax]))
            self._cab_split_actor = self.plotter.add_mesh(
                pv.Line(line_a, line_b), color="white", line_width=3,
                name="cab_split_line", reset_camera=False,
            )
            # Left sub-cuboid
            lhu = 0.5 * (split_u - umin)
            lhv = 0.5 * (vmax - vmin)
            if lhu > 1e-9:
                lc = self._plane_uv_to_world(np.array([0.5 * (umin + split_u), v_center]))
                self._cab_left_cuboid = OrientedCuboid(
                    center=lc + self.extrude_sign * half_n * n,
                    rotation=np.column_stack([self.plane_u, self.plane_v, n]),
                    extents=np.array([lhu, lhv, half_n], dtype=float),
                )
                self._cab_left_box_actor = self._draw_cuboid_wire(self._cab_left_cuboid, "orange", "cab_left_box")
            # Right sub-cuboid
            rhu = 0.5 * (umax - split_u)
            if rhu > 1e-9:
                rc = self._plane_uv_to_world(np.array([0.5 * (split_u + umax), v_center]))
                self._cab_right_cuboid = OrientedCuboid(
                    center=rc + self.extrude_sign * half_n * n,
                    rotation=np.column_stack([self.plane_u, self.plane_v, n]),
                    extents=np.array([rhu, lhv, half_n], dtype=float),
                )
                self._cab_right_box_actor = self._draw_cuboid_wire(self._cab_right_cuboid, "cyan", "cab_right_box")

        else:  # axis == "v"
            # Horizontal split line: runs left to right at split_v
            split_v = float(np.clip(self._cab_split_v, vmin + 1e-6, vmax - 1e-6))
            line_a = self._plane_uv_to_world(np.array([umin, split_v]))
            line_b = self._plane_uv_to_world(np.array([umax, split_v]))
            self._cab_split_actor = self.plotter.add_mesh(
                pv.Line(line_a, line_b), color="white", line_width=3,
                name="cab_split_line", reset_camera=False,
            )
            full_hu = 0.5 * (umax - umin)
            # Bottom sub-cuboid ("left")
            bhv = 0.5 * (split_v - vmin)
            if bhv > 1e-9:
                bc = self._plane_uv_to_world(np.array([u_center, 0.5 * (vmin + split_v)]))
                self._cab_left_cuboid = OrientedCuboid(
                    center=bc + self.extrude_sign * half_n * n,
                    rotation=np.column_stack([self.plane_u, self.plane_v, n]),
                    extents=np.array([full_hu, bhv, half_n], dtype=float),
                )
                self._cab_left_box_actor = self._draw_cuboid_wire(self._cab_left_cuboid, "orange", "cab_left_box")
            # Top sub-cuboid ("right")
            thv = 0.5 * (vmax - split_v)
            if thv > 1e-9:
                tc = self._plane_uv_to_world(np.array([u_center, 0.5 * (split_v + vmax)]))
                self._cab_right_cuboid = OrientedCuboid(
                    center=tc + self.extrude_sign * half_n * n,
                    rotation=np.column_stack([self.plane_u, self.plane_v, n]),
                    extents=np.array([full_hu, thv, half_n], dtype=float),
                )
                self._cab_right_box_actor = self._draw_cuboid_wire(self._cab_right_cuboid, "cyan", "cab_right_box")

    def _cab_enter_split_mode(self) -> None:
        self._cab_state = "split"
        self._cab_split_u = None
        self._cab_split_v = None
        self._cab_snap_pt_uv = None
        self._cab_left_edge = None
        self._cab_right_edge = None
        self._cab_left_limits = None
        self._cab_right_limits = None
        for attr in ("_cab_left_edge_actor", "_cab_right_edge_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)
        self._btn_cab_left.setEnabled(False)
        self._btn_cab_right.setEnabled(False)
        self._btn_add_cabinet.setEnabled(False)
        self._lbl_cab_edges.setText("Left: none  |  Right: none")
        self._update_cabinet_preview()
        self.plotter.render()
        self._lbl_cab_step.setText("Step 2: Click a point on any face edge to set split line.")
        self._set_status("Cabinet: click on a face edge — top/bottom = vertical split, left/right = horizontal.")

    def _cab_side_label(self, side: str) -> str:
        """Human label for a side, accounting for vertical vs horizontal split."""
        if self._cab_split_axis == "v":
            return "Bottom" if side == "left" else "Top"
        return "Left" if side == "left" else "Right"

    def _cab_arm_hinge(self, side: str) -> None:
        self._cab_hinge_target = side
        label = self._cab_side_label(side)
        btn = self._btn_cab_left if side == "left" else self._btn_cab_right
        btn.setText(f"→ click near {label} edge…")
        self._lbl_cab_step.setText(f"Click near a {label} cuboid edge.")
        self._set_status(f"Cabinet: click near the {label.lower()} sub-cuboid edge to set hinge.")

    def _pick_cabinet_edge(self, side: str) -> None:
        cuboid = self._cab_left_cuboid if side == "left" else self._cab_right_cuboid
        if cuboid is None or self.last_pick_world is None:
            self._set_status(f"[warn] Cabinet {side} cuboid not ready.")
            return

        half_u, half_v, half_n = cuboid.extents
        corners_local = np.array([
            [-half_u, -half_v, -half_n], [ half_u, -half_v, -half_n],
            [ half_u,  half_v, -half_n], [-half_u,  half_v, -half_n],
            [-half_u, -half_v,  half_n], [ half_u, -half_v,  half_n],
            [ half_u,  half_v,  half_n], [-half_u,  half_v,  half_n],
        ], dtype=float)
        corners_world = cuboid.local_to_world(corners_local)
        edge_pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        _depth_edges = {(0, 4), (1, 5), (2, 6), (3, 7)}
        _DEPTH_BONUS = 0.35

        click_uv = self._world_to_plane_uv(self.last_pick_world)
        best_ep, best_dist = edge_pairs[0], float("inf")
        for ep in edge_pairs:
            p0_uv = self._world_to_plane_uv(corners_world[ep[0]])
            p1_uv = self._world_to_plane_uv(corners_world[ep[1]])
            seg_uv = p1_uv - p0_uv
            sq = float(np.dot(seg_uv, seg_uv))
            if sq > 1e-12:
                t = float(np.clip(np.dot(click_uv - p0_uv, seg_uv) / sq, 0, 1))
                proj_uv = p0_uv + t * seg_uv
            else:
                proj_uv = p0_uv
            dist = float(np.linalg.norm(click_uv - proj_uv))
            if ep in _depth_edges:
                dist *= _DEPTH_BONUS
            if dist < best_dist:
                best_dist = dist
                best_ep = ep

        lower_ui = self._w_lower.value()
        upper_ui = self._w_upper.value()
        if lower_ui >= upper_ui:
            self._set_status("[error] Lower limit must be strictly less than upper limit.")
            return

        limits = JointLimits(lower=np.deg2rad(lower_ui), upper=np.deg2rad(upper_ui), unit="rad")
        edge = Edge(p0_world=corners_world[best_ep[0]], p1_world=corners_world[best_ep[1]])

        if side == "left":
            self._cab_left_edge = edge
            self._cab_left_limits = limits
            actor_attr = "_cab_left_edge_actor"
            color = "lime"
            name = "cab_left_edge"
        else:
            self._cab_right_edge = edge
            self._cab_right_limits = limits
            actor_attr = "_cab_right_edge_actor"
            color = "yellow"
            name = "cab_right_edge"

        old = getattr(self, actor_attr)
        if old is not None:
            self.plotter.remove_actor(old)
        setattr(self, actor_attr, self.plotter.add_mesh(
            pv.Line(edge.p0_world, edge.p1_world),
            color=color, line_width=8, name=name, reset_camera=False,
        ))
        self.plotter.render()

        # Restore button text to reflect "set, click to reselect"
        label = self._cab_side_label(side)
        btn = self._btn_cab_left if side == "left" else self._btn_cab_right
        btn.setText(f"✓ {label} Hinge (reselect)")

        left_label  = self._cab_side_label("left")
        right_label = self._cab_side_label("right")
        left_str  = "set" if self._cab_left_edge  is not None else "none"
        right_str = "set" if self._cab_right_edge is not None else "none"
        self._lbl_cab_edges.setText(f"{left_label}: {left_str}  |  {right_label}: {right_str}")

        if self._cab_left_edge is not None and self._cab_right_edge is not None:
            self._btn_add_cabinet.setEnabled(True)
            self._lbl_cab_step.setText("Ready — click 'Add Cabinet', or reselect a hinge above.")
            self._set_status("Cabinet: both hinges set. Click 'Add Cabinet' or reselect a hinge.")
        else:
            other_label = right_label if side == "left" else left_label
            other_btn = self._btn_cab_right if side == "left" else self._btn_cab_left
            other_btn.setText(f"Select {other_label} Hinge")
            self._set_status(f"Cabinet: {label.lower()} hinge set. Now select the {other_label.lower()} hinge.")

    def _add_cabinet(self) -> None:
        if self._cab_left_cuboid is None or self._cab_right_cuboid is None:
            self._set_status("[warn] Cabinet cuboids not fully defined.")
            return
        if self._cab_left_edge is None or self._cab_right_edge is None:
            self._set_status("[warn] Both hinges must be selected.")
            return

        existing = [e.cuboid for e in self._articulations if e.cuboid is not None]
        try:
            split_mesh_by_cuboid_clip(self.mesh_tm, self._cab_left_cuboid, existing_cuboids=existing)
            split_mesh_by_cuboid_clip(
                self.mesh_tm, self._cab_right_cuboid,
                existing_cuboids=existing + [self._cab_left_cuboid],
            )
        except ValueError as exc:
            self._set_status(f"[error] {exc}")
            return

        try:
            res_l = split_mesh_by_cuboid_clip(self.mesh_tm, self._cab_left_cuboid)
            left_mesh, _ = cut_cuboid_with_surface(
                res_l.inside_mesh, self._cab_left_cuboid, clip_loops=res_l.clip_loops or None,
            )
            left_mesh.vertices = left_mesh.vertices - np.asarray(self._cab_left_edge.p0_world)

            res_r = split_mesh_by_cuboid_clip(self.mesh_tm, self._cab_right_cuboid)
            right_mesh, _ = cut_cuboid_with_surface(
                res_r.inside_mesh, self._cab_right_cuboid, clip_loops=res_r.clip_loops or None,
            )
            right_mesh.vertices = right_mesh.vertices - np.asarray(self._cab_right_edge.p0_world)
        except Exception as exc:
            self._set_status(f"[error] mesh split: {exc}")
            return

        idx = len(self._articulations)
        left_actor = self._draw_cuboid_wire(self._cab_left_cuboid, "lime", f"door_box_{idx}")
        self._articulations.append(ArticulationEntry(
            edge=self._cab_left_edge,
            joint_type="revolute",
            joint_limits=self._cab_left_limits,
            door_mesh=left_mesh,
            hinge_origin=np.asarray(self._cab_left_edge.p0_world, dtype=float),
            cuboid=self._cab_left_cuboid,
            box_actor=left_actor,
        ))

        right_actor = self._draw_cuboid_wire(self._cab_right_cuboid, "lime", f"door_box_{idx + 1}")
        self._articulations.append(ArticulationEntry(
            edge=self._cab_right_edge,
            joint_type="revolute",
            joint_limits=self._cab_right_limits,
            door_mesh=right_mesh,
            hinge_origin=np.asarray(self._cab_right_edge.p0_world, dtype=float),
            cuboid=self._cab_right_cuboid,
            box_actor=right_actor,
        ))

        self._lbl_doors.setText(f"Doors queued: {len(self._articulations)}")
        self._set_status(
            f"Cabinet added as doors {idx} & {idx + 1}. Reset face to add another."
        )
        self._reset_face()

    def _reset_cabinet(self) -> None:
        self._cab_split_u = None
        self._cab_split_v = None
        self._cab_split_axis = "u"
        self._cab_snap_pt_uv = None
        self._cab_left_cuboid = None
        self._cab_right_cuboid = None
        self._cab_left_edge = None
        self._cab_right_edge = None
        self._cab_left_limits = None
        self._cab_right_limits = None
        self._cab_state = "face"
        self._cab_hinge_target = None
        for attr in ("_cab_split_actor", "_cab_snap_pt_actor", "_cab_left_box_actor",
                     "_cab_right_box_actor", "_cab_left_edge_actor", "_cab_right_edge_actor", "box_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)
        self._btn_cab_resplit.setEnabled(False)
        self._btn_cab_left.setEnabled(False)
        self._btn_cab_right.setEnabled(False)
        self._btn_add_cabinet.setEnabled(False)
        self._btn_cab_left.setText("Select Left Hinge")
        self._btn_cab_right.setText("Select Right Hinge")
        self._lbl_cab_edges.setText("Left: none  |  Right: none")
        self._lbl_cab_step.setText("Step 1: Click twice to define face.")

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        if self._rb_cuboid.isChecked():
            mode = "cuboid"
        elif self._rb_cylinder.isChecked():
            mode = "cylinder"
        else:
            mode = "cabinet"
        if mode == self._selection_mode:
            return
        self._selection_mode = mode
        self._reset_face()

        cuboid_mode = mode == "cuboid"
        cyl_mode = mode == "cylinder"
        cab_mode = mode == "cabinet"

        self._btn_hinge.setEnabled(cuboid_mode)
        self._btn_slider_j.setEnabled(cuboid_mode)
        self._btn_axis.setEnabled(cyl_mode)
        self._btn_preview_door.setEnabled(cuboid_mode)
        self._btn_add_door.setEnabled(cuboid_mode)
        self._btn_preview_cyl.setEnabled(cyl_mode)
        self._btn_add_cyl.setEnabled(cyl_mode)
        self._limits_grp.setVisible(not cab_mode)
        self._cabinet_grp.setVisible(cab_mode)

        if cuboid_mode:
            self._set_status("Click twice on the blue plane to define a face.")
        elif cyl_mode:
            self._w_lower.set_range(-360.0, 360.0)
            self._w_upper.set_range(-360.0, 360.0)
            self._lbl_limits_unit.setText("Revolute: values in degrees")
            self._set_status("Cylinder mode: click center, then perimeter point.")
        else:
            self._w_lower.set_range(-360.0, 360.0)
            self._w_upper.set_range(-360.0, 360.0)
            self._lbl_limits_unit.setText("Revolute: values in degrees")
            self._limits_grp.setVisible(True)
            self._set_status("Cabinet mode: click twice to define face, then click split line.")

    # -----------------------------------------------------------------------
    # Cylinder helpers
    # -----------------------------------------------------------------------

    def _effective_cylinder_center_uv(self) -> np.ndarray | None:
        if self._cyl_center_uv is None:
            return None
        off = np.array([self.face_offset_u, self.face_offset_v], dtype=float)
        return self._cyl_center_uv + off

    def _update_cylinder_preview(self) -> None:
        """Rebuild cylinder wireframe from current cylinder state."""
        for attr in ("face_actor", "box_actor", "edge_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)

        self.current_cylinder = None
        self.current_edge = None
        self.current_joint_type = None
        self.current_joint_limits = None
        self._staged_child_mesh = None
        self._lbl_joint.setText("Joint: none  |  Edge: not selected")

        center_uv = self._effective_cylinder_center_uv()
        if center_uv is None:
            return

        center_world = self._plane_uv_to_world(center_uv)

        if self._cyl_radius <= 1e-9:
            # Show center marker only
            sphere = pv.Sphere(radius=self.scene_diag * 0.005, center=center_world)
            self.box_actor = self.plotter.add_mesh(sphere, color="orange", name="cyl_center", reset_camera=False)
            return

        n = self.plane_n
        half_n = 0.5 * self.depth
        cyl_center = center_world + self.extrude_sign * half_n * n

        self.current_cylinder = OrientedCylinder(
            center=cyl_center,
            axis=n,
            radius=self._cyl_radius,
            half_height=half_n,
        )

        cyl_wire = pv.Cylinder(
            center=cyl_center,
            direction=n,
            radius=self._cyl_radius,
            height=self.depth,
            resolution=32,
            capping=True,
        )
        self.box_actor = self.plotter.add_mesh(
            cyl_wire, color="red", style="wireframe", line_width=2,
            name="cylinder_preview", reset_camera=False,
        )

    def _choose_cylinder_axis(self) -> None:
        if self.current_cylinder is None:
            self._set_status("[warn] No cylinder defined yet.")
            return

        lower_ui = self._w_lower.value()
        upper_ui = self._w_upper.value()
        if lower_ui >= upper_ui:
            self._set_status("[error] Lower limit must be strictly less than upper limit.")
            return
        lower = np.deg2rad(lower_ui)
        upper = np.deg2rad(upper_ui)
        limits_str = f"[{lower_ui:.1f}°, {upper_ui:.1f}°]"

        c = self.current_cylinder
        p0 = c.center - c.half_height * c.axis
        p1 = c.center + c.half_height * c.axis
        edge = Edge(p0_world=p0, p1_world=p1)
        self.current_edge = edge
        self.current_joint_type = "revolute"
        self.current_joint_limits = JointLimits(lower=lower, upper=upper, unit="rad")
        self._lbl_joint.setText(f"Joint: revolute  |  Limits: {limits_str}")
        if self.edge_actor is not None:
            self.plotter.remove_actor(self.edge_actor)
        self.edge_actor = self.plotter.add_mesh(
            pv.Line(p0, p1), color="lime", line_width=8, name="edge_selection", reset_camera=False,
        )
        self.plotter.render()
        self._set_status(f"Cylinder axis selected as revolute joint. Limits: {limits_str}")

    def _flip_extrusion_direction(self) -> None:
        self.extrude_sign *= -1.0
        had_joint = self.current_joint_type is not None
        if self._selection_mode == "cylinder":
            self._update_cylinder_preview()
        else:
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
        self._face_center_uv = None
        self._w_face_u.set_value(0.0)
        self._w_face_v.set_value(0.0)
        self.current_cuboid = None
        self.current_cylinder = None
        self._cyl_center_uv = None
        self._cyl_radius = 0.0
        self.current_edge = None
        self.current_joint_type = None
        self.current_joint_limits = None
        self.last_pick_world = None
        self._staged_child_mesh = None
        self._joint_armed = None
        self._btn_hinge.setText("Select Hinge (Revolute)")
        self._btn_slider_j.setText("Select Slider (Prismatic)")
        self._lbl_joint.setText("Joint: none  |  Edge: not selected")

        for attr in ("face_actor", "box_actor", "edge_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, attr, None)

        if self._selection_mode == "cabinet":
            self._reset_cabinet()

        self.plotter.render()
        if self._selection_mode == "cabinet":
            self._set_status("Cabinet reset. Click twice to define a new face.")
        else:
            self._set_status("Face reset. Click twice on the blue plane to define a new face.")

    def _choose_edge_joint(self, joint_type: str) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.last_pick_world is None:
            self._set_status("[warn] Click near a cuboid edge first, then select joint type.")
            return

        # Find nearest edge to last click using UV-projected distance.
        # Depth edges (parallel to plane normal) project to a single UV point,
        # so they're given a 0.35x bonus to make them win over face edges when
        # clicking near a cuboid corner — which is the typical hinge location.
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
        _depth_edges = {(0, 4), (1, 5), (2, 6), (3, 7)}
        _DEPTH_BONUS = 0.35

        click_uv = self._world_to_plane_uv(self.last_pick_world)
        best_ep, best_dist = edge_pairs[0], float("inf")
        for ep in edge_pairs:
            p0_uv = self._world_to_plane_uv(corners_world[ep[0]])
            p1_uv = self._world_to_plane_uv(corners_world[ep[1]])
            seg_uv = p1_uv - p0_uv
            sq = float(np.dot(seg_uv, seg_uv))
            if sq > 1e-12:
                t = float(np.clip(np.dot(click_uv - p0_uv, seg_uv) / sq, 0, 1))
                proj_uv = p0_uv + t * seg_uv
            else:
                proj_uv = p0_uv
            dist = float(np.linalg.norm(click_uv - proj_uv))
            if ep in _depth_edges:
                dist *= _DEPTH_BONUS
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
            edge_line, color="lime", line_width=8, name="edge_selection", reset_camera=False,
        )
        self.plotter.render()
        if joint_type == "revolute":
            self._btn_hinge.setText("✓ Hinge (reselect)")
            self._btn_slider_j.setText("Select Slider (Prismatic)")
        else:
            self._btn_slider_j.setText("✓ Slider (reselect)")
            self._btn_hinge.setText("Select Hinge (Revolute)")
        self._set_status(f"[{joint_type}] edge selected. Limits: {limits_str}")

    def _choose_hinge(self) -> None:
        self._w_lower.set_range(-360.0, 360.0)
        self._w_upper.set_range(-360.0, 360.0)
        self._lbl_limits_unit.setText("Revolute: values in degrees")
        self._arm_joint("revolute")

    def _choose_slider(self) -> None:
        d = self.scene_diag
        self._w_lower.set_range(-2 * d, 2 * d)
        self._w_upper.set_range(-2 * d, 2 * d)
        self._lbl_limits_unit.setText("Prismatic: values in meters")
        self._arm_joint("prismatic")

    def _arm_joint(self, joint_type: str) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        self._joint_armed = joint_type
        if joint_type == "revolute":
            self._btn_hinge.setText("→ click near edge…")
            self._btn_slider_j.setText("Select Slider (Prismatic)")
        else:
            self._btn_slider_j.setText("→ click near edge…")
            self._btn_hinge.setText("Select Hinge (Revolute)")
        self._set_status(f"Armed {joint_type} joint — click near a cuboid edge.")

    def _print_current_cuboid(self) -> None:
        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        print("\nCurrent cuboid:")
        print(f"center = np.array({self.current_cuboid.center.tolist()})")
        print(f"rotation = np.array({self.current_cuboid.rotation.tolist()})")
        print(f"extents = np.array({self.current_cuboid.extents.tolist()})\n")
        self._set_status("Cuboid info printed to console.")

    def _apply_cylinder(self) -> None:
        self._apply_door()

    def _add_cylinder(self) -> None:
        self._add_door()

    def _apply_door(self) -> None:
        if self._selection_mode == "cylinder":
            if self.current_cylinder is None:
                self._set_status("[warn] No cylinder defined yet.")
                return
            if self.current_edge is None:
                self._set_status("[warn] Select axis joint first.")
                return
            try:
                result = split_mesh_by_cylinder_clip(self.mesh_tm, self.current_cylinder)
                hinge = np.asarray(self.current_edge.p0_world, dtype=float)
                door_mesh, _ = cut_cylinder_with_surface(
                    result.inside_mesh, self.current_cylinder,
                    clip_loops=result.clip_loops or None,
                )
                door_mesh.vertices = door_mesh.vertices - hinge
                self._staged_child_mesh = door_mesh
                self._set_status(
                    f"Cylinder door preview — {len(door_mesh.faces)} faces. "
                    f"Click 'Add Door' to queue it."
                )
            except Exception as exc:
                self._set_status(f"[error] door: {exc}")
            return

        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.current_edge is None:
            self._set_status("[warn] Select joint type first.")
            return
        try:
            result = split_mesh_by_cuboid_clip(self.mesh_tm, self.current_cuboid)
            hinge = np.asarray(self.current_edge.p0_world, dtype=float)
            door_mesh, _ = cut_cuboid_with_surface(
                result.inside_mesh, self.current_cuboid,
                clip_loops=result.clip_loops or None,
            )
            door_mesh.vertices = door_mesh.vertices - hinge
            self._staged_child_mesh = door_mesh
            self._set_status(
                f"Door preview — {len(door_mesh.faces)} faces. "
                f"Click 'Add Door' to queue it."
            )
        except Exception as exc:
            self._set_status(f"[error] door: {exc}")

    def _add_door(self) -> None:
        if self._selection_mode == "cylinder":
            if self.current_cylinder is None:
                self._set_status("[warn] No cylinder defined yet.")
                return
            if self.current_edge is None or self.current_joint_type is None or self.current_joint_limits is None:
                self._set_status("[warn] Select axis joint first.")
                return

            # Overlap check against all existing selections
            centroids = self.mesh_tm.triangles_center
            new_inside = self.current_cylinder.contains(centroids)
            for i, e in enumerate(self._articulations):
                other = e.cuboid.contains(centroids) if e.cuboid is not None else e.cylinder.contains(centroids)
                if (new_inside & other).any():
                    self._set_status(f"[error] Cylinder overlaps existing selection {i}.")
                    return

            if self._staged_child_mesh is None:
                self._apply_door()
                if self._staged_child_mesh is None:
                    return

            cyl = self.current_cylinder
            actor = self.plotter.add_mesh(
                pv.Cylinder(
                    center=cyl.center, direction=cyl.axis,
                    radius=cyl.radius, height=2 * cyl.half_height,
                    resolution=32, capping=True,
                ),
                color="lime", style="wireframe", line_width=2,
                name=f"door_cyl_{len(self._articulations)}", reset_camera=False,
            )
            entry = ArticulationEntry(
                edge=self.current_edge,
                joint_type=self.current_joint_type,
                joint_limits=self.current_joint_limits,
                door_mesh=self._staged_child_mesh,
                hinge_origin=np.asarray(self.current_edge.p0_world, dtype=float),
                cylinder=self.current_cylinder,
                box_actor=actor,
            )
            self._articulations.append(entry)
            self._lbl_doors.setText(f"Doors queued: {len(self._articulations)}")
            self._set_status(
                f"Cylinder {len(self._articulations) - 1} added "
                f"({len(self._staged_child_mesh.faces)} faces). Reset to add another."
            )
            self._reset_face()
            return

        if self.current_cuboid is None:
            self._set_status("[warn] No cuboid defined yet.")
            return
        if self.current_edge is None or self.current_joint_type is None or self.current_joint_limits is None:
            self._set_status("[warn] Select joint type and limits first.")
            return

        # Check for overlap with already-queued cuboids (ignore cylinder entries)
        existing = [e.cuboid for e in self._articulations if e.cuboid is not None]
        try:
            split_mesh_by_cuboid_clip(self.mesh_tm, self.current_cuboid, existing_cuboids=existing)
        except ValueError as exc:
            self._set_status(f"[error] {exc}")
            return

        # Run door computation if not already previewed
        if self._staged_child_mesh is None:
            self._apply_door()
            if self._staged_child_mesh is None:
                return  # _apply_door already set error status

        # Draw a persistent green wireframe for this door's cuboid
        half_u, half_v, half_n = self.current_cuboid.extents
        corners_local = np.array([
            [-half_u, -half_v, -half_n], [ half_u, -half_v, -half_n],
            [ half_u,  half_v, -half_n], [-half_u,  half_v, -half_n],
            [-half_u, -half_v,  half_n], [ half_u, -half_v,  half_n],
            [ half_u,  half_v,  half_n], [-half_u,  half_v,  half_n],
        ], dtype=float)
        corners_world = self.current_cuboid.local_to_world(corners_local)
        edge_pairs = [
            (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)
        ]
        lines: list[int] = []
        for a, b in edge_pairs:
            lines.extend([2, a, b])
        wire = pv.PolyData()
        wire.points = corners_world
        wire.lines = np.array(lines, dtype=np.int32)
        actor = self.plotter.add_mesh(
            wire, color="lime", line_width=2,
            name=f"door_box_{len(self._articulations)}", reset_camera=False,
        )

        entry = ArticulationEntry(
            edge=self.current_edge,
            joint_type=self.current_joint_type,
            joint_limits=self.current_joint_limits,
            door_mesh=self._staged_child_mesh,
            hinge_origin=np.asarray(self.current_edge.p0_world, dtype=float),
            cuboid=self.current_cuboid,
            box_actor=actor,
        )
        self._articulations.append(entry)
        self._lbl_doors.setText(f"Doors queued: {len(self._articulations)}")
        self._set_status(
            f"Door {len(self._articulations) - 1} added "
            f"({len(self._staged_child_mesh.faces)} faces). Reset face to add another."
        )
        # Clear current selection so user can start a new one
        self._reset_face()

    def _clear_doors(self) -> None:
        for entry in self._articulations:
            if entry.box_actor is not None:
                self.plotter.remove_actor(entry.box_actor)
        self._articulations.clear()
        self._lbl_doors.setText("Doors queued: 0")
        self.plotter.render()
        self._set_status("All queued doors cleared.")

    def _create_urdf_file(self) -> None:
        if not self._articulations:
            self._set_status("[warn] No doors queued. Add at least one door first.")
            return

        export_dir = Path("data/output/urdf_test")
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute remainder by sequentially removing each door from the original mesh
            remainder = self.mesh_tm.copy()
            for entry in self._articulations:
                if entry.cuboid is not None:
                    clip_result = split_mesh_by_cuboid_clip(remainder, entry.cuboid)
                else:
                    clip_result = split_mesh_by_cylinder_clip(remainder, entry.cylinder)
                remainder = clip_result.outside_mesh

            if self._chk_slice_only.isChecked():
                # Export base + doors as separate objects in one GLB file
                scene = trimesh.Scene()
                scene.add_geometry(remainder, node_name="base", geom_name="base")
                for i, entry in enumerate(self._articulations):
                    m = entry.door_mesh.copy()
                    m.vertices = m.vertices + entry.hinge_origin
                    scene.add_geometry(m, node_name=f"door_{i}", geom_name=f"door_{i}")
                out_path = export_dir / "sliced_combined.glb"
                scene.export(out_path)
                self._set_status(
                    f"Sliced scene saved to {out_path}  "
                    f"({len(self._articulations)} door(s) + base, separate objects)"
                )
                return

            base_stl = export_dir / "base.stl"
            remainder.export(base_stl)

            # Save each door mesh and build ArticulationSpec list
            specs: list[ArticulationSpec] = []
            for i, entry in enumerate(self._articulations):
                door_stl = export_dir / f"door_{i}.stl"
                entry.door_mesh.export(door_stl)
                specs.append(ArticulationSpec(
                    child_mesh_stl=door_stl,
                    edge=entry.edge,
                    joint_type=entry.joint_type,
                    joint_limits=entry.joint_limits,
                    link_name=f"door_{i}",
                ))

            urdf_path = export_dir / "name.urdf"
            export_to_urdf(
                urdf_path=urdf_path,
                parent_mesh_stl=base_stl,
                articulations=specs,
            )
            self._set_status(
                f"URDF saved to {urdf_path}  "
                f"({len(specs)} door(s), base + {len(specs)} STL(s))"
            )
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
    window = CuboidSelectorApp("data/input/microwave.stl")
    window.run()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
