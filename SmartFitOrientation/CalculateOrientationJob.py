# Copyright (c) 2022 Jaime van Kessel
# SmartFit Orientation plugin. Licensed under LGPL-3.0. See LICENSE for the full license text.

from UM.Job import Job
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.RotateOperation import RotateOperation
from UM.Operations.TranslateOperation import TranslateOperation
from cura.CuraApplication import CuraApplication
from .MeshTweaker import Tweak
from UM.Math.Quaternion import Quaternion
from UM.Math.Vector import Vector
from UM.Scene.SceneNode import SceneNode
import math
import numpy as np

from typing import List, TYPE_CHECKING, Optional, Tuple, Any

if TYPE_CHECKING:
    from UM.Message import Message

# Cura world space: Y is up.
#   index 0 = X  (machine_width)
#   index 1 = Y  (machine_height, vertical)
#   index 2 = Z  (machine_depth)
_X = 0
_Y = 1
_Z = 2

# Sweep resolution: every N degrees we test a new tilt.
_SWEEP_STEP_DEG = 3


class CalculateOrientationJob(Job):
    def __init__(self, nodes: List[SceneNode], extended_mode: bool = False, message: Optional["Message"] = None) -> None:
        super().__init__()
        self._message = message
        self._nodes = nodes
        self._extended_mode = extended_mode
        self._skipped_nodes = 0

    def run(self) -> None:
        build_volume = self._getBuildVolumeSize()

        # --- Step 1: find best orientation for each node and rotate ---
        rotation_op = GroupedOperation()
        rotated_nodes = []

        for node in self._nodes:
            transformed_vertices = node.getMeshDataTransformed().getVertices()
            vertices = self._toNumpyVertices(transformed_vertices)
            triangles = self._toTriangles(vertices)

            quaternion = self._pickBestOrientation(vertices, triangles, build_volume, transformed_vertices)

            if quaternion is None:
                self._skipped_nodes += 1
                Job.yieldThread()
                continue

            rotation_center = node.getBoundingBox().center
            rotation_op.addOperation(RotateOperation(node, quaternion, rotate_around_point=rotation_center))
            rotated_nodes.append((node, build_volume))
            Job.yieldThread()

        rotation_op.push()

        # --- Step 2: snap to bed + nudge inside build plate using actual post-rotation bounds ---
        # Using Cura's real bounding box avoids all coordinate system guessing.
        placement_op = GroupedOperation()
        for node, bv in rotated_nodes:
            bb = node.getBoundingBox()
            if bb is None:
                continue

            # Y is up: bottom of model must sit at Y = 0 (the bed)
            offset_y = -bb.minimum.y
            offset_x, offset_z = 0.0, 0.0

            if bv is not None:
                machine_width, machine_depth, _ = bv
                x0, x1, z0, z1 = self._getBuildPlateBounds(machine_width, machine_depth)
                offset_x = self._clampToRange(bb.minimum.x, bb.maximum.x, x0, x1)
                offset_z = self._clampToRange(bb.minimum.z, bb.maximum.z, z0, z1)

            if abs(offset_x) > 1e-4 or abs(offset_y) > 1e-4 or abs(offset_z) > 1e-4:
                placement_op.addOperation(TranslateOperation(node, Vector(offset_x, offset_y, offset_z)))

        placement_op.push()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _pickBestOrientation(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        build_volume: Optional[Tuple[float, float, float]],
        raw_vertices: Any
    ) -> Optional[Quaternion]:
        """
        Algorithm:
          1. Generate all candidate orientations (MeshTweaker + full grid sweep).
          2. Score every candidate by estimated support material needed.
          3. Sort candidates from least to most support.
          4. Walk down the sorted list:
             - For each candidate, try all yaw angles (spin on bed) to fit the build volume.
             - The first candidate that fits is chosen.
          5. If build volume is unknown, just return the single best-support orientation.
        """
        centered = vertices - np.mean(vertices, axis=0)
        sampled = self._sampleVertices(centered, 1200)

        # --- collect candidates ---
        scored: List[Tuple[float, np.ndarray]] = []  # (support_score, rotation_matrix)

        # MeshTweaker candidates
        tweak_result = Tweak(
            raw_vertices,
            extended_mode=self._extended_mode,
            verbose=False,
            progress_callback=self.updateProgress,
            min_volume=CuraApplication.getInstance().getPreferences().getValue("SmartFitOrientation/min_volume")
        )
        for candidate in (getattr(tweak_result, "best_5", None) or []):
            v, phi = candidate[5][0], candidate[5][1]
            mat = self._buildRotationMatrix(v, phi)
            score = self._supportScore(triangles, mat)
            scored.append((score, mat))

        # Grid sweep — tilt around X then Z, no yaw yet (yaw is searched per candidate below)
        for ax_deg in range(0, 181, _SWEEP_STEP_DEG):
            ax = math.radians(ax_deg)
            Rx = self._aamat([1.0, 0.0, 0.0], ax)
            for az_deg in range(0, 181, _SWEEP_STEP_DEG):
                az = math.radians(az_deg)
                Rz = self._aamat([0.0, 0.0, 1.0], az)
                mat = np.matmul(Rz, Rx)   # apply Rx first, then Rz
                score = self._supportScore(triangles, mat)
                scored.append((score, mat))

            Job.yieldThread()   # keep Cura responsive during the outer loop

        # --- sort by support score (lowest = least support = best) ---
        scored.sort(key=lambda t: t[0])

        if build_volume is None:
            # No machine configured — just take the best support orientation
            return self._matrixToQuaternion(scored[0][1]) if scored else None

        machine_width, machine_depth, machine_height = build_volume
        eps = 0.5

        # --- walk sorted list, pick first that fits ---
        for score, mat in scored:
            rotated = np.matmul(sampled, mat.T)

            # Height must fit (Y extent in Cura space)
            if np.max(rotated[:, _Y]) - np.min(rotated[:, _Y]) > machine_height + eps:
                continue

            # Try yaw angles to fit the XZ footprint
            yaw = self._findYaw(rotated, machine_width, machine_depth, eps, 0.5)
            if yaw is None:
                continue

            # Found a fitting orientation — build and return the quaternion
            if yaw != 0:
                Ry = self._aamat([0.0, 1.0, 0.0], yaw)
                mat = np.matmul(Ry, mat)

            return self._matrixToQuaternion(mat)

        return None   # nothing fits — model is physically too large to rotate into volume

    # ------------------------------------------------------------------
    # Rotation helpers
    # ------------------------------------------------------------------

    def _buildRotationMatrix(self, v: List[float], phi: float) -> np.ndarray:
        """MeshTweaker euler params → 3×3 rotation matrix in Cura Y-up space."""
        return np.matmul(
            self._aamat([1.0, 0.0, 0.0], -0.5 * math.pi),   # Z-up → Y-up conversion
            self._aamat([-v[0], -v[1], -v[2]], phi)
        )

    def _aamat(self, axis: List[float], angle: float) -> np.ndarray:
        """Axis-angle → 3×3 rotation matrix."""
        a = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(a)
        if n == 0:
            return np.identity(3)
        x, y, z = a / n
        c, s, ic = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
        return np.array([
            [c + x*x*ic,   x*y*ic - z*s, x*z*ic + y*s],
            [y*x*ic + z*s, c + y*y*ic,   y*z*ic - x*s],
            [z*x*ic - y*s, z*y*ic + x*s, c + z*z*ic  ]
        ], dtype=np.float64)

    def _matrixToQuaternion(self, mat: np.ndarray) -> Quaternion:
        """Convert 3×3 rotation matrix to a Cura Quaternion."""
        # Standard matrix → quaternion (Shepperd method)
        m = mat
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return Quaternion(w, x, y, z)

    # ------------------------------------------------------------------
    # Fit checking (Cura Y-up: X=width, Y=height, Z=depth)
    # ------------------------------------------------------------------

    def _findYaw(
        self,
        vertices: np.ndarray,
        machine_width: float,
        machine_depth: float,
        eps: float,
        step_deg: float
    ) -> Optional[float]:
        """Find the smallest yaw (rotation around Y) that fits the XZ footprint."""
        x = vertices[:, _X]
        z = vertices[:, _Z]
        if np.max(x) - np.min(x) <= machine_width + eps and np.max(z) - np.min(z) <= machine_depth + eps:
            return 0.0
        steps = int(round(180.0 / step_deg))
        for i in range(1, steps + 1):
            yaw = math.radians(i * step_deg)
            c, s = math.cos(yaw), math.sin(yaw)
            rx = x * c - z * s
            rz = x * s + z * c
            if np.max(rx) - np.min(rx) <= machine_width + eps and np.max(rz) - np.min(rz) <= machine_depth + eps:
                return yaw
        return None

    # ------------------------------------------------------------------
    # Support scoring (Y-up: Y index is vertical)
    # ------------------------------------------------------------------

    def _supportScore(self, triangles: np.ndarray, rotation_matrix: np.ndarray) -> float:
        """Lower = less support needed = better orientation."""
        if len(triangles) == 0:
            return 0.0

        rotated = np.matmul(triangles, rotation_matrix.T)
        ea = rotated[:, 1, :] - rotated[:, 0, :]
        eb = rotated[:, 2, :] - rotated[:, 0, :]
        normals = np.cross(ea, eb)
        lengths = np.linalg.norm(normals, axis=1)
        mask = lengths > 1e-8
        if not np.any(mask):
            return 0.0

        unit_n = normals[mask] / lengths[mask, np.newaxis]
        areas = lengths[mask] * 0.5
        y_vals = rotated[mask][:, :, _Y]
        y_min = float(np.min(y_vals))
        tri_max_y = np.max(y_vals, axis=1)

        first_layer = 0.2
        ascent = -0.07809801382985776  # from MeshTweaker PARAMETER

        bottom_mask = tri_max_y <= y_min + first_layer
        bottom_area = float(np.sum(areas[bottom_mask])) if np.any(bottom_mask) else 0.0

        overhang_mask = (unit_n[:, _Y] < ascent) & (tri_max_y > y_min + first_layer)
        if np.any(overhang_mask):
            inner = unit_n[overhang_mask, _Y] - ascent
            overhang_area = float(2.0 * np.sum(areas[overhang_mask] * np.abs(inner * (inner < 0)) ** 2))
        else:
            overhang_area = 0.0

        return overhang_area - 0.15 * bottom_area

    # ------------------------------------------------------------------
    # Build volume / placement helpers
    # ------------------------------------------------------------------

    def _getBuildVolumeSize(self) -> Optional[Tuple[float, float, float]]:
        stack = CuraApplication.getInstance().getGlobalContainerStack()
        if not stack:
            return None
        try:
            return (
                float(stack.getProperty("machine_width", "value")),
                float(stack.getProperty("machine_depth", "value")),
                float(stack.getProperty("machine_height", "value"))
            )
        except (TypeError, ValueError):
            return None

    def _getBuildPlateBounds(self, machine_width: float, machine_depth: float) -> Tuple[float, float, float, float]:
        stack = CuraApplication.getInstance().getGlobalContainerStack()
        center_is_zero = False
        if stack:
            val = stack.getProperty("machine_center_is_zero", "value")
            center_is_zero = val.lower() == "true" if isinstance(val, str) else bool(val)
        if center_is_zero:
            return -machine_width / 2, machine_width / 2, -machine_depth / 2, machine_depth / 2
        return 0.0, machine_width, 0.0, machine_depth

    def _clampToRange(self, obj_min: float, obj_max: float, lim_min: float, lim_max: float) -> float:
        """Return the translation offset needed to move [obj_min, obj_max] inside [lim_min, lim_max]."""
        if obj_min < lim_min:
            return lim_min - obj_min
        if obj_max > lim_max:
            return lim_max - obj_max
        return 0.0

    # ------------------------------------------------------------------
    # Vertex helpers
    # ------------------------------------------------------------------

    def _toNumpyVertices(self, vertices: Any) -> np.ndarray:
        arr = np.asarray(vertices, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        out = []
        for v in vertices:
            out.append([v.x, v.y, v.z] if hasattr(v, "x") else list(v))
        return np.asarray(out, dtype=np.float64)

    def _toTriangles(self, vertices: np.ndarray) -> np.ndarray:
        n = len(vertices) // 3
        if n == 0:
            return np.empty((0, 3, 3), dtype=np.float64)
        return vertices[:n * 3].reshape(n, 3, 3)

    def _sampleVertices(self, vertices: np.ndarray, max_count: int) -> np.ndarray:
        if len(vertices) <= max_count:
            return vertices
        return vertices[::int(math.ceil(len(vertices) / max_count))]

    # ------------------------------------------------------------------
    # Progress / message
    # ------------------------------------------------------------------

    def updateProgress(self, progress):
        if self._message:
            self._message.setProgress(progress)

    def getMessage(self) -> Optional["Message"]:
        return self._message

    def getSkippedNodeCount(self) -> int:
        return self._skipped_nodes
