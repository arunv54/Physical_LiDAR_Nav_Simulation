"""Saved-data playback keeps measurement chronology and full-map statistics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lidar_project.visualization.report_data import _mapping_payload


class ReportDataTests(unittest.TestCase):
    def saved_run(self, directory):
        directory = Path(directory)
        (directory / "mapping.json").write_text(json.dumps({
            "extent_m": [0, 2, 0, 2], "voxel_size_m": .12, "resolution_m": 1.,
            "map_point_count": 4, "observed_cell_fraction": .75}))
        (directory / "trajectory.json").write_text(json.dumps({
            "name": "baseline", "status": "goal", "seed": 1,
            "config": {"mount_height_m": .6, "robot": {
                "v_max": 1., "omega_max_deg_s": 90., "radius_m": .35, "height_m": 1.}},
            "sensor": {}, "world_objects": []}))
        np.savez(directory / "mapping.npz", map_xyz=[[.1, .1, 0.], [.2, .2, 1.], [1.1, .2, .3], [1.2, 1.2, .4]],
                 map_first_seen=[0, 1, 1, 1], ground_first_seen=[[0, -1], [-1, -1]],
                 obstacle_first_seen=[[1, 1], [-1, 1]])
        np.savez(directory / "trajectory.npz", states_t=[0., 1., 2.], controls_t=[0., 1.],
                 states_xy=[[0., 0.], [1., 0.], [2., 0.]], states_theta=[0., 0., 0.],
                 reference_xy=[[0., 0.], [2., 0.]], controls_v=[1., 1.], controls_omega=[0., 0.],
                 controls_n_returns=[3, 5], controls_n_obstacle_points=[1, 2],
                 controls_swept_surface_gap=[.5, .5])
        return directory

    def test_display_sampling_preserves_first_seen_and_full_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = _mapping_payload(self.saved_run(temporary), max_points=2)
        self.assertEqual(result["full_map_points"], 4)
        self.assertEqual(result["display_map_points"], 2)
        self.assertEqual(result["points"], [[.1, .1, 0., 0], [1.2, 1.2, .4, 1]])
        self.assertEqual([row["points"] for row in result["full_map_curve"]], [1, 4])
        self.assertEqual([row["observed_fraction"] for row in result["full_map_curve"]], [.25, .75])
        # Scrubbing back filters the same immutable first-seen records.
        self.assertEqual([p for p in result["points"] if p[3] <= 0], [[.1, .1, 0., 0]])
        self.assertEqual(result["map_step_at_state"], [0, 1, 1])
        self.assertEqual(result["scan_times"], [0., 1.])
        self.assertEqual(result["times"], [0., 1., 2.])
        self.assertEqual(result["motion_check"]["max_recorded_euler_position_residual_m"], 0.)

    def test_invalid_future_measurement_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.saved_run(temporary)
            path = directory / "mapping.npz"
            with np.load(path) as arrays:
                values = {key: arrays[key] for key in arrays.files}
            values["map_first_seen"] = [0, 1, 1, 2]
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "first-observation"):
                _mapping_payload(directory)


if __name__ == "__main__":
    unittest.main()
