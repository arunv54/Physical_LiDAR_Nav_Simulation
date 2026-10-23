"""Causal invariants and honest measurement denominators for the factorial study."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lidar_project import api
from lidar_project.experiments.navigation_dependencies import (
    TARGET_LABEL, apply_case, emitted_rays, scan_diagnostics, study_config,
)
from lidar_project.experiments.sampling_reflectance import (
    BRIGHT_REFLECTANCE, DARK_REFLECTANCE, _replay_recording, controlled_invariants,
    distance_bins, factorial_cases, factorial_contrasts,
)
from lidar_project.trajectory import generate_reference, load_result, run_trajectory


class SamplingReflectance(unittest.TestCase):
    def test_full_factorial_preserves_mount_and_scene_except_diffuse_coefficient(self):
        baseline = study_config()
        cases = factorial_cases()
        configs = [apply_case(baseline, case) for case in cases]
        fixed = controlled_invariants(configs)
        self.assertEqual(len({(c['azimuth_samples'], c['target_diffuse_reflectance']) for c in cases}), 4)
        self.assertEqual(fixed['mount_height_m'], .6)
        self.assertEqual(fixed['sensor_mount']['yaw_deg'], 0.)
        self.assertEqual(fixed['sensor_mount']['pitch_deg'], 0.)
        changed = copy.deepcopy(configs)
        changed[1]['sensor_mount']['yaw_deg'] = 90.
        with self.assertRaises(ValueError):
            controlled_invariants(changed)
        changed = copy.deepcopy(configs)
        target = next(obj for obj in changed[1]['world']['unexpected_objects'] if obj['label'] == TARGET_LABEL)
        target['material']['rho_s'] = .1
        with self.assertRaises(ValueError):
            controlled_invariants(changed)

    def test_sampling_changes_emitted_count_not_fov_or_power(self):
        pose = api.SensorPose(position=(0., 0., .6))
        dense = api.LidarConfig(n_channels=8, measurements_per_cycle=360, n_beam_samples=1)
        sparse = api.replace(dense, measurements_per_cycle=90)
        origin_a, direction_a = emitted_rays(dense, pose)
        origin_b, direction_b = emitted_rays(sparse, pose)
        self.assertEqual(len(direction_a), 4 * len(direction_b))
        np.testing.assert_allclose(direction_a.reshape(360, 8, 3)[::4], direction_b.reshape(90, 8, 3), atol=1e-14)
        self.assertEqual(dense.peak_power_w, sparse.peak_power_w)

    def test_target_power_scales_with_reflectance_before_threshold_selection(self):
        cfg = api.LidarConfig(n_channels=1, vfov_upper_deg=0., measurements_per_cycle=1,
                              hfov_start_deg=0., hfov_end_deg=.01, n_beam_samples=1,
                              noise_enabled=False, min_range_m=.5, max_range_m=20.)
        pose = api.SensorPose(position=(0., 0., .5))
        diagnostics = []
        for rho in (BRIGHT_REFLECTANCE, DARK_REFLECTANCE):
            target = api.OrientedBox((5., 0., .5), (1., 2., 1.),
                                     material=api.Material('virtual_diffuse', rho), label=TARGET_LABEL)
            scene = api.Scene([target])
            cloud = api.LidarSensor(cfg, pose, seed=1).scan(scene)
            diagnostics.append(scan_diagnostics(cloud, pose, scene, cfg,
                               {'controller': {'z_ground_m': .12}, 'robot': {'height_m': 1.}}))
        self.assertEqual(diagnostics[0]['eligible_echoes'], diagnostics[1]['eligible_echoes'])
        ratio = diagnostics[0]['mean_eligible_power_w'] / diagnostics[1]['mean_eligible_power_w']
        self.assertAlmostEqual(ratio, BRIGHT_REFLECTANCE / DARK_REFLECTANCE, places=9)

    def test_fixed_angle_sensor_translation_changes_power_and_detection_probability(self):
        cfg = api.LidarConfig(n_channels=1, vfov_upper_deg=0., measurements_per_cycle=1,
                              hfov_start_deg=0., hfov_end_deg=.01, n_beam_samples=1,
                              noise_enabled=True, min_range_m=.5, max_range_m=20.)
        target = api.OrientedBox((5., 0., .5), (1., 2., 1.),
                                 material=api.Material('weak_virtual_diffuse', DARK_REFLECTANCE),
                                 label=TARGET_LABEL)
        scene = api.Scene([target])
        rows = []
        for position in ((0., 0., .5), (3., 0., .5)):
            pose = api.SensorPose(position=position)
            scan = api.LidarSensor(cfg, pose, seed=1).scan(scene)
            rows.append(scan_diagnostics(scan, pose, scene, cfg,
                        {'controller': {'z_ground_m': .12}, 'robot': {'height_m': 1.}}))
        self.assertEqual(rows[0]['eligible_echoes'], rows[1]['eligible_echoes'])
        self.assertGreater(rows[1]['mean_eligible_power_w'], rows[0]['mean_eligible_power_w'])
        self.assertGreater(rows[1]['expected_detection_probability'], rows[0]['expected_detection_probability'])
        binned = distance_bins(rows)
        self.assertEqual(sum(row['eligible_echoes'] for row in binned), 2)
        self.assertEqual(len(binned), 2)
        self.assertLess(binned[0]['mean_eligible_range_m'], binned[1]['mean_eligible_range_m'])

    def test_replay_retains_every_saved_pose_and_scan_target_identity(self):
        config = study_config()
        config['robot']['max_steps'] = 2
        source = run_trajectory(config, 'baseline', seed=1, reference=generate_reference(config))
        case = factorial_cases()[3]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            metrics, traces = _replay_recording(apply_case(config, case), case, 1, source, destination)
            loaded = load_result(destination)
            self.assertEqual(loaded['status'], 'replay')
            np.testing.assert_array_equal(loaded['states']['xy'], source['states']['xy'])
            np.testing.assert_array_equal(loaded['states']['theta'], source['states']['theta'])
            self.assertEqual(len(loaded['scans']), len(source['controls']['t']))
            with np.load(destination / 'scan_targets.npz', allow_pickle=False) as masks:
                self.assertEqual(masks['scan_offsets'][-1], sum(len(scan['xyz_world']) for scan in loaded['scans']))
                self.assertEqual(int(masks['target_mask'].sum()), metrics['detected_returns'])
            self.assertTrue((destination / 'traces.csv').exists())
            self.assertEqual(sum(r['detected_returns'] for r in traces), metrics['detected_returns'])

    def test_factorial_contrasts_keep_sampling_and_reflectance_comparisons_separate(self):
        metrics = dict(mean_returns_per_scan=100., mean_target_returns_per_scan=10.,
                       mean_target_eligible_per_scan=20., observed_detection_probability=.5,
                       expected_detection_probability=.5, mean_eligible_power_w=1e-7,
                       range_rmse_m=.02, observed_fraction=.3)
        rows = []
        for case, offset in zip(factorial_cases(), (3, 2, 1, 0)):
            rows.append(dict(metrics, case_id=case['id'], seed=1,
                             mean_target_returns_per_scan=metrics['mean_target_returns_per_scan'] + offset))
        contrasts = factorial_contrasts(rows)
        self.assertEqual(contrasts[0]['name'], 'sampling_at_bright')
        self.assertEqual(contrasts[0]['mean_paired_difference']['mean_target_returns_per_scan'], 2.)
        self.assertEqual(contrasts[2]['name'], 'reflectance_at_dense')
        self.assertEqual(contrasts[2]['mean_paired_difference']['mean_target_returns_per_scan'], 1.)


if __name__ == '__main__':
    unittest.main()
