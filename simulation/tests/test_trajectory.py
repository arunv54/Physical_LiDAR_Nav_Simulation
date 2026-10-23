"""Focused checks of the new route/tracking experiment, not legacy navigation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from lidar_project import api
from lidar_project.trajectory import (
    generate_reference, load_config, load_result, run_trajectory, save_result,
    swept_surface_gap,
)


class TrajectoryGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.reference = generate_reference(cls.config)

    def test_reference_retains_checked_corners_and_omits_unexpected_map(self):
        ref, cfg = self.reference, self.config
        known = api.build_scene({"objects": cfg["world"]["known_objects"]})
        for point in ref["raw_xy"]:
            self.assertLess(np.linalg.norm(ref["xy"] - point, axis=1).min(), 1e-12)
        for a, b in zip(ref["xy"][:-1], ref["xy"][1:]):
            self.assertGreater(swept_surface_gap(a, b, known, ref["inflation_m"], cfg["robot"]["height_m"]), -1e-10)
        self.assertAlmostEqual(np.linalg.norm(np.diff(ref["xy"], axis=0), axis=1).sum(), ref["s"][-1])
        changed = copy.deepcopy(cfg)
        changed["world"]["unexpected_objects"] = []
        np.testing.assert_array_equal(generate_reference(changed)["xy"], ref["xy"])

    def test_reference_time_is_nominal_cruise_not_corner_dynamics(self):
        ref = self.reference
        np.testing.assert_allclose(ref["t"], ref["s"] / self.config["planner"]["reference_speed_mps"])
        self.assertEqual(ref["speed"][-1], 0.0)
        self.assertIn("Nominal", ref["timing"])

    def test_swept_scoring_catches_between_sample_collision(self):
        obstacle = api.OrientedBox((0, 0, 0.5), (0.02, 1, 1))
        a, b = np.array([-1., 0]), np.array([1., 0])
        self.assertGreater(swept_surface_gap(a, a, [obstacle], 0.1, 1), 0)
        self.assertGreater(swept_surface_gap(b, b, [obstacle], 0.1, 1), 0)
        self.assertAlmostEqual(swept_surface_gap(a, b, [obstacle], 0.1, 1), -0.1)
        overhead = api.OrientedBox((0, 0, 3), (1, 1, 1))
        self.assertTrue(np.isinf(swept_surface_gap(a, b, [overhead], 0.1, 1)))
        cylinder = api.VerticalCylinder((0, 0, 0), 0.5, 1)
        self.assertAlmostEqual(swept_surface_gap([-1, 2], [1, 2], [cylinder], 0.2, 1), 1.3)

    def test_final_state_and_control_limits(self):
        cfg = copy.deepcopy(self.config)
        cfg["robot"]["max_steps"] = 1
        result = run_trajectory(cfg, "baseline", seed=3, reference=self.reference)
        self.assertEqual(len(result["states"]["t"]), 2)
        self.assertEqual(len(result["controls"]["t"]), 1)
        self.assertAlmostEqual(result["states"]["t"][-1], cfg["robot"]["dt"])
        expected = result["states"]["xy"][0] + cfg["robot"]["dt"] * result["controls"]["v"][0] * np.array([
            np.cos(result["states"]["theta"][0]), np.sin(result["states"]["theta"][0])])
        np.testing.assert_allclose(result["states"]["xy"][-1], expected)
        self.assertLessEqual(abs(result["controls"]["omega"][0]), np.radians(cfg["robot"]["omega_max_deg_s"]))
        self.assertLessEqual(result["controls"]["v"][0], cfg["robot"]["v_max"])

    def test_recording_does_not_change_seeded_control(self):
        cfg = copy.deepcopy(self.config)
        cfg["robot"]["max_steps"] = 25
        original_scan = api.LidarSensor.scan
        captured, recorded_calls, plain_calls = [], [], []
        def capture(sensor, scene):
            cloud = original_scan(sensor, scene)
            captured.append((copy.deepcopy(sensor.rng.bit_generator.state),
                             cloud.xyz_world.copy(), cloud.intensity.copy()))
            return cloud
        with mock.patch.object(api.LidarSensor, "scan", new=capture):
            captured = recorded_calls
            recorded = run_trajectory(cfg, "baseline", seed=17, record_history=True, reference=self.reference)
            captured = plain_calls
            plain = run_trajectory(cfg, "baseline", seed=17, record_history=False, reference=self.reference)
        for key in recorded["states"]:
            np.testing.assert_array_equal(recorded["states"][key], plain["states"][key])
        for key in recorded["controls"]:
            np.testing.assert_array_equal(recorded["controls"][key], plain["controls"][key])
        self.assertEqual(len(recorded["observations"]), len(recorded["controls"]["t"]))
        self.assertEqual(plain["observations"], [])
        self.assertTrue(recorded["scans_available"])
        self.assertFalse(plain["scans_available"])
        self.assertEqual(plain["scans"], [])
        self.assertEqual(len(recorded_calls), len(recorded["controls"]["t"]))
        self.assertEqual(len(plain_calls), len(recorded_calls))
        for scan, source, unrecorded in zip(recorded["scans"], recorded_calls, plain_calls):
            self.assertEqual(source[0], unrecorded[0])
            np.testing.assert_array_equal(source[1], unrecorded[1])
            np.testing.assert_array_equal(source[2], unrecorded[2])
            np.testing.assert_array_equal(scan["xyz_world"], source[1])
            np.testing.assert_array_equal(scan["intensity"], source[2])
        self.assertTrue(np.any(recorded["scans"][0]["xyz_world"][:, 2] < cfg["controller"]["z_ground_m"]))

    def test_baseline_reaches_goal_and_lidar_disabled_hits_withheld_obstacle(self):
        baseline = run_trajectory(self.config, "baseline", seed=1, reference=self.reference)
        disabled = run_trajectory(self.config, {"name": "disabled", "lidar_enabled": False}, seed=1,
                                  record_history=False, reference=self.reference)
        self.assertEqual(baseline["status"], "goal")
        self.assertGreater(baseline["controls"]["swept_surface_gap"].min(), 0)
        self.assertLessEqual(np.linalg.norm(baseline["states"]["xy"][-1] - self.config["robot"]["goal"]),
                             self.config["robot"]["goal_tolerance_m"])
        self.assertEqual(disabled["status"], "collision")
        unknown = self.config["world"]["unexpected_objects"]
        self.assertLess(swept_surface_gap(disabled["states"]["xy"][-2], disabled["states"]["xy"][-1], unknown,
                                         self.config["robot"]["radius_m"], self.config["robot"]["height_m"]), 0)
        self.assertTrue(np.all(disabled["controls"]["n_returns"] == 0))

    def test_initial_unknown_overlap_and_lossless_portable_archive(self):
        cfg = copy.deepcopy(self.config)
        cfg["world"]["unexpected_objects"][0]["base"] = [0, 0, 0]
        overlap = run_trajectory(cfg, "baseline", reference=self.reference)
        self.assertEqual(overlap["status"], "collision")
        self.assertEqual(len(overlap["states"]["t"]), 1)
        self.assertEqual(len(overlap["controls"]["t"]), 0)
        cfg = copy.deepcopy(self.config)
        cfg["robot"]["max_steps"] = 3
        original = run_trajectory(cfg, "baseline", reference=self.reference)
        with tempfile.TemporaryDirectory() as temporary:
            source = save_result(original, Path(temporary) / "original")
            moved = source.rename(Path(temporary) / "relocated")
            restored = load_result(moved)
            self.assertEqual(restored["status"], original["status"])
            for group in ("states", "controls"):
                for key in original[group]:
                    np.testing.assert_array_equal(original[group][key], restored[group][key])
            for a, b in zip(original["observations"], restored["observations"]):
                np.testing.assert_array_equal(a, b)
            self.assertTrue(restored["scans_available"])
            self.assertEqual(len(restored["scans"]), len(original["controls"]["t"]))
            for a, b in zip(original["scans"], restored["scans"]):
                np.testing.assert_array_equal(a["xyz_world"], b["xyz_world"])
                np.testing.assert_array_equal(a["intensity"], b["intensity"])
            with np.load(moved / "trajectory.npz", allow_pickle=False) as archive:
                self.assertFalse(any(archive[key].dtype.hasobject for key in archive.files))

    def test_disabled_snapshots_and_no_recording_remain_distinct(self):
        cfg = copy.deepcopy(self.config)
        cfg["robot"]["max_steps"] = 3
        disabled = run_trajectory(cfg, {"name": "disabled", "lidar_enabled": False},
                                  reference=self.reference, record_history=True)
        self.assertTrue(disabled["scans_available"])
        self.assertEqual(len(disabled["scans"]), 3)
        for scan in disabled["scans"]:
            self.assertEqual(scan["xyz_world"].shape, (0, 3))
            self.assertEqual(scan["intensity"].shape, (0,))
        unrecorded = run_trajectory(cfg, "baseline", reference=self.reference, record_history=False)
        with tempfile.TemporaryDirectory() as temporary:
            restored = load_result(save_result(disabled, Path(temporary) / "disabled"))
            self.assertTrue(restored["scans_available"])
            self.assertEqual(len(restored["scans"]), 3)
            missing = load_result(save_result(unrecorded, Path(temporary) / "unrecorded"))
            self.assertFalse(missing["scans_available"])
            self.assertEqual(missing["scans"], [])

    def test_v1_archive_keeps_raw_scans_explicitly_unavailable(self):
        cfg = copy.deepcopy(self.config)
        cfg["robot"]["max_steps"] = 2
        original = run_trajectory(cfg, "baseline", reference=self.reference)
        with tempfile.TemporaryDirectory() as temporary:
            folder = save_result(original, temporary)
            # Reproduce the legacy schema by removing all v2-only fields.
            metadata = json.loads((folder / "trajectory.json").read_text())
            metadata["format_version"] = 1
            metadata.pop("scans_available")
            (folder / "trajectory.json").write_text(json.dumps(metadata))
            with np.load(folder / "trajectory.npz", allow_pickle=False) as archive:
                old = {key: archive[key].copy() for key in archive.files if not key.startswith("scan_")}
            np.savez_compressed(folder / "trajectory.npz", **old)
            restored = load_result(folder)
            self.assertFalse(restored["scans_available"])
            self.assertEqual(restored["scans"], [])
            self.assertTrue(restored["history_recorded"])
            for key in original["states"]:
                np.testing.assert_array_equal(restored["states"][key], original["states"][key])
            for a, b in zip(restored["observations"], original["observations"]):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
