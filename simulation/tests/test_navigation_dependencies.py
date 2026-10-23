"""Check independent eligibility denominators and failure-aware aggregation."""
import unittest

import numpy as np

from lidar_project import api
from lidar_project.experiments.navigation_dependencies import (
    TARGET_LABEL, aggregate_navigation, apply_case, dependency_cases, opportunity_audit,
    pool_diagnostics, scan_diagnostics, study_config,
)


class NavigationDependencies(unittest.TestCase):
    def setUp(self):
        self.cfg = api.LidarConfig(n_channels=1, vfov_upper_deg=0., measurements_per_cycle=1,
                                   hfov_start_deg=0., hfov_end_deg=.01, n_beam_samples=1,
                                   noise_enabled=False, min_range_m=.5, max_range_m=20.)
        self.pose = api.SensorPose(position=(0., 0., .5))
        self.target = api.OrientedBox((5., 0., .5), (1., 2., 1.), material=api.Material("diffuse", .3), label=TARGET_LABEL)
        self.config = {"controller": {"z_ground_m": .12}, "robot": {"height_m": 1.}}

    def test_los_and_range_gate_have_different_denominators(self):
        scene = api.Scene([self.target])
        clear = opportunity_audit(self.cfg, self.pose, scene)
        self.assertEqual(clear["target_opportunities"], 1)
        self.assertEqual(clear["visible_target_echoes"], 1)
        self.assertEqual(clear["eligible_echoes"], 1)
        self.assertEqual(clear["full_fov_opportunities"], 1)
        blocker = api.OrientedBox((.2, 0., .5), (.1, 2., 1.), label="below_min_opaque")
        blocked = opportunity_audit(self.cfg, self.pose, api.Scene([blocker, self.target]))
        self.assertEqual(blocked["target_opportunities"], 1)
        self.assertEqual(blocked["blocked_target_echoes"], 1)
        self.assertEqual(blocked["eligible_echoes"], 0)
        gated = opportunity_audit(api.replace(self.cfg, max_range_m=4.), self.pose, scene)
        self.assertEqual(gated["visible_target_echoes"], 1)
        self.assertEqual(gated["eligible_echoes"], 0)

    def test_target_conditioning_uses_echo_truth_and_independent_power(self):
        scene = api.Scene([self.target])
        pc = api.LidarSensor(self.cfg, self.pose, seed=7).scan(scene)
        row = scan_diagnostics(pc, self.pose, scene, self.cfg, self.config)
        self.assertEqual(row["detected_returns"], 1)
        self.assertEqual(row["target_controller_points"], 1)
        self.assertEqual(row["observed_detection_probability"], 1.)
        self.assertEqual(row["range_rmse_m"], 0.)
        self.assertAlmostEqual(row["eligible_power_sum_w"], row["power_sum_w"], places=15)
        pooled = pool_diagnostics([row, row])
        self.assertEqual(pooled["eligible_echoes"], 2)
        self.assertEqual(pooled["observed_detection_probability"], 1.)

    def test_dark_target_is_eligible_even_when_undetected(self):
        self.target.material = api.Material("almost_black", 1e-12)
        scene = api.Scene([self.target])
        pc = api.LidarSensor(self.cfg, self.pose, seed=7).scan(scene)
        row = scan_diagnostics(pc, self.pose, scene, self.cfg, self.config)
        self.assertEqual(row["eligible_echoes"], 1)
        self.assertEqual(row["detected_returns"], 0)
        self.assertEqual(row["expected_detections"], 0)
        self.assertIsNone(row["range_rmse_m"])

    def test_cases_preserve_known_map_and_fov_yaw_has_matched_baseline(self):
        config = study_config()
        cases = dependency_cases()
        for case in cases:
            active = apply_case(config, case)
            self.assertEqual(active["world"]["known_objects"], config["world"]["known_objects"])
            self.assertEqual(active["robot"], config["robot"])
            self.assertEqual(active["controller"], config["controller"])
        side = next(c for c in cases if c["id"] == "fov_side")
        self.assertEqual(side["comparison_baseline"], "fov_front")

    def test_failure_prefix_is_retained_in_summary(self):
        shared = dict(goal_completed=1, collision=0, timeout=0, elapsed_s=10., path_length_m=12.,
                      swept_min_surface_gap_m=.2, observed_fraction=.6, spatial_route_rmse_m=.3,
                      mean_returns_per_control_step=100.)
        failure = {**shared, "goal_completed": 0, "collision": 1, "elapsed_s": 3.,
                   "path_length_m": 2., "swept_min_surface_gap_m": -.1}
        summary = aggregate_navigation([shared, failure])
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["goals"], 1)
        self.assertEqual(summary["collisions"], 1)
        self.assertAlmostEqual(summary["mean_elapsed_s"], 6.5)
        self.assertAlmostEqual(summary["worst_surface_gap_m"], -.1)


if __name__ == "__main__":
    unittest.main()
