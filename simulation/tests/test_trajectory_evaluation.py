"""Checks for the study's timing/spatial distinction and failure denominators."""
import unittest
from pathlib import Path

import numpy as np

from lidar_project.experiments.trajectory import (
    evaluate_result, point_polyline_distances, tracking_metrics,
)


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_spatial_and_clock_errors_are_different(self):
        reference = {"xy": np.array([[0., 0.], [2., 0.]]), "t": np.array([0., 1.])}
        positions = np.array([[0., 0.], [1., 0.], [2., 0.]])
        metrics = tracking_metrics(np.array([0., 1., 2.]), positions, reference)
        self.assertEqual(metrics["spatial_route_rmse_m"], 0.)
        self.assertAlmostEqual(metrics["time_reference_rmse_m"], 1 / np.sqrt(3))
        self.assertEqual(metrics["time_reference_max_m"], 1.)

    def test_terminal_state_contributes_to_error(self):
        reference = {"xy": np.array([[0., 0.], [2., 0.]]), "t": np.array([0., 2.])}
        positions = np.array([[0., 0.], [1., 0.], [2., 3.]])
        metrics = tracking_metrics(np.array([0., 1., 2.]), positions, reference)
        self.assertAlmostEqual(metrics["spatial_route_rmse_m"], np.sqrt(3))
        self.assertEqual(metrics["spatial_route_max_m"], 3.)
        self.assertAlmostEqual(metrics["time_reference_rmse_m"], np.sqrt(3))

    def test_distance_uses_segments_not_only_waypoints(self):
        distances = point_polyline_distances([[1., 1.], [3., 0.]], [[0., 0.], [2., 0.]])
        np.testing.assert_array_equal(distances, [1., 1.])



    def _simple_execution(self, obstacle_x):
        from lidar_project.trajectory import load_config, run_trajectory
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "trajectory.jsonc")
        config["robot"].update(start=[0., 0.], goal=[3., 0.], start_heading_deg=0.,
                               v_max=10., dt=.2, max_steps=1)
        config["world"]["known_objects"] = []
        config["world"]["unexpected_objects"] = [
            {"type": "box", "label": "test_obstacle", "center": [obstacle_x, 0., .5],
             "size": [.2, .2, 1.], "material": "concrete"}]
        xy = np.array([[0., 0.], [3., 0.]])
        reference = dict(xy=xy, raw_xy=xy.copy(), s=np.array([0., 3.]), t=np.array([0., 3.75]),
                         speed=np.array([.8, .8]), heading=np.zeros(2))
        return run_trajectory(config, {"name": "test", "lidar_enabled": False},
                              seed=1, record_history=False, reference=reference)

    def test_initial_collision_has_one_state_and_no_controls(self):
        result = self._simple_execution(0.)
        metric = evaluate_result(result, "test", "initial_overlap", 1, lidar_enabled=False)
        self.assertEqual(metric["status"], "collision")
        self.assertEqual(metric["collision"], 1)
        self.assertEqual(metric["state_count"], 1)
        self.assertEqual(metric["control_steps"], 0)
        self.assertEqual(metric["elapsed_s"], 0.)
        self.assertLess(metric["initial_surface_gap_m"], 0.)

    def test_swept_collision_counts_when_endpoints_are_clear(self):
        result = self._simple_execution(1.)
        self.assertTrue(np.all(result["states"]["surface_gap"] > 0))
        metric = evaluate_result(result, "test", "between_states", 1, lidar_enabled=False)
        self.assertEqual(metric["status"], "collision")
        self.assertLess(metric["swept_min_surface_gap_m"], 0.)
        self.assertEqual(metric["state_count"], 2)
        self.assertEqual(metric["control_steps"], 1)
        self.assertAlmostEqual(metric["elapsed_s"], .2)
        self.assertAlmostEqual(metric["path_length_m"], 2.)


if __name__ == "__main__":
    unittest.main()
