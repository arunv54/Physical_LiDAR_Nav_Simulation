"""Rigid mounting geometry and passive scan-observer invariants."""
import copy
import unittest
from unittest import mock

import numpy as np

from lidar_project import api
from lidar_project.trajectory import generate_reference, load_config, run_trajectory
from lidar_project.trajectory.configuration import effective_config
from lidar_project.trajectory.engine import mounted_sensor_pose


class SensorMount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.config["robot"]["max_steps"] = 4
        cls.config["sensor_overrides"].update(n_channels=3, measurements_per_cycle=36)
        cls.reference = generate_reference(cls.config)

    def test_body_offset_rotates_and_euler_mount_composes(self):
        cfg = copy.deepcopy(self.config)
        cfg["sensor_mount"] = dict(offset_xy_m=[0.3, -0.2], roll_deg=5, pitch_deg=12, yaw_deg=-20)
        effective, _ = effective_config(cfg, {"name": "turned", "sensor_mount": {"pitch_deg": 17}})
        pose = mounted_sensor_pose(effective, [2, 3], np.pi / 2)
        np.testing.assert_allclose(pose.position, [2.2, 3.3, 0.6])
        self.assertAlmostEqual(pose.yaw_deg, 70)
        self.assertEqual(pose.roll_deg, 5)
        self.assertEqual(pose.pitch_deg, 17)
        self.assertEqual(cfg["sensor_mount"]["pitch_deg"], 12)

    def test_invalid_mount_values_fail_before_sensing(self):
        cases = [None, {"offset_xy_m": [1]}, {"offset_xy_m": [0, float("inf")]},
                 {"pitch_deg": float("nan")}, {"roll_deg": "invalid"}, {"x": 2}]
        for mount in cases:
            with self.subTest(mount=mount), self.assertRaises(ValueError):
                effective_config(self.config, {"sensor_mount": mount})
        for height in (float("inf"), -0.1):
            with self.subTest(height=height), self.assertRaises(ValueError):
                effective_config(self.config, {"mount_height_m": height})

    def test_explicit_zero_mount_matches_legacy_scans_controls_and_rng(self):
        original_scan = api.LidarSensor.scan
        captures = []

        def capture(sensor, scene):
            scan = original_scan(sensor, scene)
            captures[-1].append((copy.deepcopy(sensor.rng.bit_generator.state),
                                 scan.xyz_world.copy(), scan.intensity.copy()))
            return scan

        def legacy_pose(config, position, theta):
            return api.SensorPose(position=(*position, config["mount_height_m"]),
                                  yaw_deg=float(np.degrees(theta)))

        cfg = copy.deepcopy(self.config)
        cfg["sensor_mount"] = dict(offset_xy_m=[0, 0], roll_deg=0, pitch_deg=0, yaw_deg=0)
        with mock.patch.object(api.LidarSensor, "scan", new=capture):
            captures.append([])
            absent = run_trajectory(self.config, seed=17, reference=self.reference)
            captures.append([])
            zero = run_trajectory(cfg, seed=17, reference=self.reference)
            captures.append([])
            with mock.patch("lidar_project.trajectory.engine.mounted_sensor_pose", new=legacy_pose):
                legacy = run_trajectory(self.config, seed=17, reference=self.reference)
        for compared in (zero, legacy):
            for group in ("states", "controls"):
                for key in absent[group]:
                    np.testing.assert_array_equal(absent[group][key], compared[group][key])
        for compared in captures[1:]:
            self.assertEqual(len(captures[0]), len(compared))
            for old, new in zip(captures[0], compared):
                self.assertEqual(old[0], new[0])
                np.testing.assert_array_equal(old[1], new[1])
                np.testing.assert_array_equal(old[2], new[2])

    def test_pitch_changes_real_measurements_at_same_robot_state(self):
        base = run_trajectory(self.config, seed=8, reference=self.reference)
        pitched = run_trajectory(self.config, {"sensor_mount": {"pitch_deg": 18}},
                                 seed=8, reference=self.reference)
        np.testing.assert_array_equal(base["states"]["xy"][0], pitched["states"]["xy"][0])
        a, b = base["scans"][0]["xyz_world"], pitched["scans"][0]["xyz_world"]
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)
        self.assertFalse(np.array_equal(a, b))

    def test_observer_uses_existing_measurement_without_retaining_data(self):
        events = []

        def observe(**event):
            events.append(event)
            # Mutable pose/position copies must not alter the controller state.
            event["pose"].yaw_deg = 999
            event["robot_xy"][:] = 999

        with mock.patch.object(api.LidarSensor, "scan", autospec=True,
                               side_effect=api.LidarSensor.scan) as scans:
            observed = run_trajectory(self.config, seed=11, record_history=False,
                                      reference=self.reference, on_scan=observe)
        plain = run_trajectory(self.config, seed=11, record_history=False, reference=self.reference)
        self.assertEqual(scans.call_count, len(observed["controls"]["t"]))
        self.assertEqual(len(events), scans.call_count)
        self.assertEqual(observed["scans"], [])
        self.assertNotIn("diagnostics", observed)
        np.testing.assert_array_equal([event["time"] for event in events], observed["controls"]["t"])
        np.testing.assert_array_equal([event["robot_theta"] for event in events], observed["states"]["theta"][:-1])
        for group in ("states", "controls"):
            for key in plain[group]:
                np.testing.assert_array_equal(plain[group][key], observed[group][key])
        disabled_events = []
        run_trajectory(self.config, {"lidar_enabled": False}, reference=self.reference,
                       on_scan=lambda **event: disabled_events.append(event))
        self.assertEqual(disabled_events, [])


if __name__ == "__main__":
    unittest.main()
