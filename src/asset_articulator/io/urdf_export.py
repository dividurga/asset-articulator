from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np

from asset_articulator.assets.joints import JointLimits
from asset_articulator.geometry.edge import Edge


@dataclass
class ArticulationSpec:
    """One articulated child link for URDF export.

    child_mesh_stl
        Path to the child STL file, already expressed in the child's local
        frame (i.e. vertices shifted by -hinge_origin before saving).
    edge
        The selected joint edge in world coordinates.
    joint_type
        ``"revolute"`` or ``"prismatic"``.
    joint_limits
        Lower/upper limits (radians for revolute, metres for prismatic).
    link_name
        Name used in the URDF for this link/joint pair (e.g. ``"door_0"``).
    """

    child_mesh_stl: str | Path
    edge: Edge
    joint_type: str
    joint_limits: JointLimits
    link_name: str = "door"


def export_to_urdf(
    urdf_path: str | Path,
    parent_mesh_stl: str | Path,
    articulations: list[ArticulationSpec],
) -> None:
    """Export one or more articulated joints as a URDF file.

    Frame convention
    ----------------
    - Base link frame == world frame used when exporting the parent STL.
    - Each child link frame == its joint origin (edge.p0_world).
    - Each child STL is expected to already be in that local frame
      (vertices shifted by -edge.p0_world before writing).

    URDF structure
    --------------
    ::

        <robot>
          <link name="base"> ... </link>
          <link name="door_0"> ... </link>
          <joint name="joint_0" type="revolute">
            <parent link="base"/>
            <child  link="door_0"/>
            ...
          </joint>
          <link name="door_1"> ... </link>
          <joint name="joint_1" ...> ... </joint>
          ...
        </robot>
    """
    if not articulations:
        raise ValueError("At least one ArticulationSpec is required.")

    for spec in articulations:
        if spec.joint_type not in {"revolute", "prismatic", "continuous"}:
            raise ValueError(f"Unsupported joint type: {spec.joint_type!r}")

    urdf_path = Path(urdf_path)
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_dir = urdf_path.parent.resolve()

    parent_mesh_stl = Path(parent_mesh_stl).resolve()
    if not parent_mesh_stl.exists():
        raise FileNotFoundError(f"Parent mesh STL not found: {parent_mesh_stl}")
    parent_mesh_rel = Path(os.path.relpath(parent_mesh_stl, start=urdf_dir)).as_posix()

    # Colour palette: base gets grey, children cycle through these rgba strings
    _CHILD_COLOURS = [
        "0.72 0.45 0.20 1.0",  # warm wood
        "0.25 0.55 0.80 1.0",  # steel blue
        "0.30 0.70 0.40 1.0",  # sage green
        "0.80 0.35 0.30 1.0",  # terracotta
        "0.60 0.40 0.75 1.0",  # lavender
        "0.85 0.65 0.20 1.0",  # amber
    ]

    robot = ET.Element("robot", name="articulated_object")

    # Base link
    base_link = ET.SubElement(robot, "link", name="base")
    base_visual = ET.SubElement(base_link, "visual")
    ET.SubElement(base_visual, "origin", xyz="0 0 0", rpy="0 0 0")
    base_geo = ET.SubElement(base_visual, "geometry")
    ET.SubElement(base_geo, "mesh", filename=parent_mesh_rel)
    base_mat = ET.SubElement(base_visual, "material", name="base_material")
    ET.SubElement(base_mat, "color", rgba="0.75 0.75 0.75 1.0")

    for i, spec in enumerate(articulations):
        child_stl = Path(spec.child_mesh_stl).resolve()
        if not child_stl.exists():
            raise FileNotFoundError(f"Child mesh STL not found: {child_stl}")
        child_rel = Path(os.path.relpath(child_stl, start=urdf_dir)).as_posix()

        # Ensure unique link/joint names
        link_name = spec.link_name if spec.link_name != "door" else f"door_{i}"
        joint_name = f"joint_{i}"

        joint_origin = np.asarray(spec.edge.p0_world, dtype=float)
        axis = np.asarray(spec.edge.p1_world - spec.edge.p0_world, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            raise ValueError(f"Articulation {i}: edge is too short to define a valid axis.")
        axis = axis / axis_norm

        # Child link
        child_link = ET.SubElement(robot, "link", name=link_name)
        child_visual = ET.SubElement(child_link, "visual")
        ET.SubElement(child_visual, "origin", xyz="0 0 0", rpy="0 0 0")
        child_geo = ET.SubElement(child_visual, "geometry")
        ET.SubElement(child_geo, "mesh", filename=child_rel)
        child_mat = ET.SubElement(child_visual, "material", name=f"child_material_{i}")
        ET.SubElement(child_mat, "color", rgba=_CHILD_COLOURS[i % len(_CHILD_COLOURS)])

        # Joint
        joint = ET.SubElement(robot, "joint", name=joint_name, type=spec.joint_type)
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=link_name)
        ET.SubElement(joint, "origin", xyz=_fmt(joint_origin), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=_fmt(axis))
        if spec.joint_type != "continuous":
            ET.SubElement(
                joint, "limit",
                lower=str(spec.joint_limits.lower),
                upper=str(spec.joint_limits.upper),
                effort="100.0",
                velocity="1.0",
            )

    urdf_path.write_text(_prettify(robot), encoding="utf-8")


def _fmt(v: np.ndarray) -> str:
    return f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"


def _prettify(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")
