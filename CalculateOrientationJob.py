# Copyright (c) 2022 Jaime van Kessel
# SmartFit Orientation plugin. Licensed under LGPL-3.0. See LICENSE for the full license text.

from UM.Job import Job
from UM.Logger import Logger
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

from typing import List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from UM.Message import Message


class CalculateOrientationJob(Job):
    def __init__(self, nodes: List[SceneNode], extended_mode: bool = False, message: Optional["Message"] = None) -> None:
        super().__init__()
        self._message = message
        self._nodes = nodes
        self._extended_mode = extended_mode

    def run(self) -> None:
        op = GroupedOperation()

        build_volume = None
        global_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if global_stack:
            try:
                bw = global_stack.getProperty("machine_width", "value")
                bd = global_stack.getProperty("machine_depth", "value")
                bh = global_stack.getProperty("machine_height", "value")
                if bw and bd and bh:
                    build_volume = (float(bw), float(bd), float(bh))
                    Logger.log('d', 'SmartFitOrientation: build volume read from printer profile: '
                               '{:.1f} x {:.1f} x {:.1f} mm (W x D x H)'.format(*build_volume))
                else:
                    Logger.log('w', 'SmartFitOrientation: could not read build volume '
                               '(got width={}, depth={}, height={})'.format(bw, bd, bh))
            except Exception as e:
                Logger.log('e', 'SmartFitOrientation: exception reading build volume: {}'.format(e))

        for node in self._nodes:
            transformed_vertices = node.getMeshDataTransformed().getVertices()

            prefs = CuraApplication.getInstance().getPreferences()
            min_vol_val = prefs.getValue("SmartFitOrientation/min_volume")
            min_volume = str(min_vol_val).lower() != "false" if min_vol_val is not None else True
            fast_fit_val = prefs.getValue("SmartFitOrientation/fast_fit_check")
            fast_fit_check = str(fast_fit_val).lower() != "false" if fast_fit_val is not None else True
            result = Tweak(transformed_vertices, extended_mode=self._extended_mode, verbose=False, progress_callback=self.updateProgress, min_volume=min_volume, build_volume=build_volume, fast_fit_check=fast_fit_check)

            [v, phi] = result.euler_parameter

            # Convert the new orientation into quaternion
            new_orientation = Quaternion.fromAngleAxis(phi, Vector(-v[0], -v[1], -v[2]))
            # Rotate the axis frame (Z-up MeshTweaker space → Y-up Cura world space)
            new_orientation = Quaternion.fromAngleAxis(-0.5 * math.pi, Vector(1, 0, 0)) * new_orientation

            bbox = node.getBoundingBox()
            rotate_center = bbox.center
            center_arr = np.array([rotate_center.x, rotate_center.y, rotate_center.z])

            # Apply the base euler rotation to the vertices so we can work in
            # final Cura world space for the spin search and drop-to-plate step.
            base_rot = new_orientation.toMatrix().getData()[:3, :3]
            post_euler_verts = np.dot(transformed_vertices - center_arr, base_rot.T) + center_arr

            # ── Spin search ──────────────────────────────────────────────────
            # The fit check in MeshTweaker may have found the orientation fits
            # only at a specific in-plane rotation angle.  We now find that
            # angle directly in Cura world space (Y is up, so "spin on the bed"
            # = rotation around the Y axis) and fold it into new_orientation.
            spin_deg = 0.0
            if build_volume is not None:
                bw, bd, _ = build_volume
                bf_min, bf_max = sorted([bw, bd])
                px = post_euler_verts[:, 0]
                pz = post_euler_verts[:, 2]
                best_fit_fp_max = float('inf')
                best_fit_spin   = None
                for step in range(360):      # 0.0°, 0.5°, 1.0°, … 179.5°
                    deg = step * 0.5
                    a = math.radians(deg)
                    ca, sa = math.cos(a), math.sin(a)
                    # Standard Y-axis rotation: x' = cos·x + sin·z,  z' = −sin·x + cos·z
                    # Must match Quaternion.fromAngleAxis(deg, Y).toMatrix() convention.
                    eu = float(np.max(ca * px + sa * pz) - np.min(ca * px + sa * pz))
                    ew = float(np.max(-sa * px + ca * pz) - np.min(-sa * px + ca * pz))
                    fp_min = min(eu, ew)
                    fp_max = max(eu, ew)
                    if fp_min <= bf_min and fp_max <= bf_max:
                        if fp_max < best_fit_fp_max:
                            best_fit_fp_max = fp_max
                            best_fit_spin   = (deg, fp_min, fp_max)

                if best_fit_spin is not None:
                    spin_deg, found_fp_min, found_fp_max = best_fit_spin
                    Logger.log('d',
                        'SmartFitOrientation: spin search — best fitting angle {:.1f}° '
                        '→ footprint {:.1f}x{:.1f} fits in {:.0f}x{:.0f}'.format(
                            spin_deg, found_fp_min, found_fp_max, bf_min, bf_max))

            if spin_deg != 0.0:
                spin = Quaternion.fromAngleAxis(math.radians(spin_deg), Vector(0, 1, 0))
                new_orientation = spin * new_orientation
                # Recompute vertex positions with the spin included
                full_rot = new_orientation.toMatrix().getData()[:3, :3]
                final_verts = np.dot(transformed_vertices - center_arr, full_rot.T) + center_arr
            else:
                final_verts = post_euler_verts

            # Ensure node gets the new orientation, rotated around the center of
            # the object (prevents weird position jumps on the build plate).
            op.addOperation(RotateOperation(node, new_orientation, rotate_around_point=rotate_center))

            # ── Drop to build plate ──────────────────────────────────────────
            # After rotation the bounding-box centre stays fixed but the model's
            # bottom may no longer sit on the build plate (Y=0).
            try:
                new_min_y = float(np.min(final_verts[:, 1]))
                if abs(new_min_y) > 0.01:
                    op.addOperation(TranslateOperation(node, Vector(0, -new_min_y, 0)))

                rv_min = np.min(final_verts, axis=0)
                rv_max = np.max(final_verts, axis=0)
                Logger.log('i',
                    'SmartFitOrientation: final model bounding box: '
                    'W={:.1f}  H={:.1f}  D={:.1f}  '
                    '(build volume W={} H={} D={})  spin={:.0f}°'.format(
                        float(rv_max[0] - rv_min[0]),
                        float(rv_max[1] - rv_min[1]),
                        float(rv_max[2] - rv_min[2]),
                        build_volume[0] if build_volume else '?',
                        build_volume[1] if build_volume else '?',
                        build_volume[2] if build_volume else '?',
                        spin_deg))
            except Exception as e:
                Logger.log('e', 'SmartFitOrientation: exception in drop-to-plate step: {}'.format(e))

            # ── Centre on build plate ────────────────────────────────────────
            # After a large rotation the model's XZ centre may be off the plate
            # even though its dimensions fit.  Use the build volume's own world-
            # space bounding box so this works for both centred and corner-origin
            # printers automatically.
            try:
                bv_bbox = CuraApplication.getInstance().getBuildVolume().getBoundingBox()
                if bv_bbox is not None:
                    plate_cx = float(bv_bbox.minimum.x + bv_bbox.maximum.x) / 2.0
                    plate_cz = float(bv_bbox.minimum.z + bv_bbox.maximum.z) / 2.0
                    model_cx = float(rv_min[0] + rv_max[0]) / 2.0
                    model_cz = float(rv_min[2] + rv_max[2]) / 2.0
                    dx = plate_cx - model_cx
                    dz = plate_cz - model_cz
                    if abs(dx) > 0.1 or abs(dz) > 0.1:
                        Logger.log('d', 'SmartFitOrientation: centring on build plate '
                                   'Δx={:.1f}  Δz={:.1f}'.format(dx, dz))
                        op.addOperation(TranslateOperation(node, Vector(dx, 0.0, dz)))
            except Exception as e:
                Logger.log('e', 'SmartFitOrientation: exception in centre-on-plate step: {}'.format(e))

            Job.yieldThread()
        op.push()

    def updateProgress(self, progress):
        if self._message:
            self._message.setProgress(progress)

    def getMessage(self) -> Optional["Message"]:
        return self._message