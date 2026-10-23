"""Protect threshold denominators, original-ray extraction and trial provenance."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lidar_project import api
from lidar_project.experiments.navigation_threshold import (
    expanded_power, pool_settings, prepare_design, run_study, summarize_trials,
    wilson_interval,
)


class NavigationThreshold(unittest.TestCase):
    def setUp(self):
        self.cfg = api.LidarConfig(n_channels=1, vfov_upper_deg=0., n_beam_samples=1,
                                   measurements_per_cycle=1, hfov_start_deg=0., hfov_end_deg=.01)

    def source(self, directory):
        directory = Path(directory)
        config = dict(robot={}, controller={}, mount_height_m=.5,
                      sensor_mount=dict(offset_xy_m=[0., 0.], roll_deg=0., pitch_deg=0., yaw_deg=0.),
                      _sensor_base_snapshot=asdict(self.cfg),
                      world=dict(known_objects=[], unexpected_objects=[
                          dict(type='box', label='study_target', center=[10., 0., .5],
                               size=[.8, 2., 1.], yaw_deg=0., material=dict(rho_d=.3, rho_s=0.))]))
        (directory/'manifest.json').write_text(json.dumps(dict(baseline_config=config)))
        np.savez_compressed(directory/'reference_poses_seed_1.npz', t=np.arange(3.),
                            xy=np.array([[0., 0.], [3., 0.], [6., 0.]]), theta=np.zeros(3))
        return directory

    def test_misses_are_trials_but_not_zero_range_errors(self):
        row = summarize_trials([False, False], [np.nan, np.nan], [np.nan, np.nan], self.cfg, .2)
        self.assertEqual(row['trials'], 2)
        self.assertEqual(row['detections'], 0)
        self.assertEqual(row['observed_pd'], 0.)
        self.assertAlmostEqual(row['brier_score'], .04)
        self.assertIsNone(row['range_rmse_m'])
        self.assertGreater(row['wilson_high'], 0.)
        low, high = wilson_interval(2, 2)
        self.assertLess(low, 1.)
        self.assertEqual(high, 1.)

    def test_pool_uses_counts_and_detected_only_range_variance(self):
        first = summarize_trials([True], [.1], [10.], self.cfg, .5)
        second = summarize_trials([False, False, False], [np.nan]*3, [np.nan]*3, self.cfg, .2)
        pool = pool_settings([first, second])
        self.assertEqual(pool['trials'], 4)
        self.assertEqual(pool['detections'], 1)
        self.assertAlmostEqual(pool['conditional_range_rmse_m'], .1)
        self.assertAlmostEqual(pool['calibration_rmse'], np.sqrt((.5**2+3*.2**2)/4))
        self.assertAlmostEqual(pool['mean_brier_score'], (.25+3*.04)/4)

    def test_design_keeps_real_navigation_ray_and_independent_optical_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            design, scene = prepare_design(self.source(temp))
            self.assertEqual(len(design['views']), 3)
            self.assertEqual(len({v['step'] for v in design['views']}), 3)
            self.assertEqual(len(design['settings']), 33)
            for setting in design['settings']:
                cfg = api.LidarConfig(**setting['sensor_config'])
                view = design['views'][setting['pose_id']]
                cfg = api.replace(cfg, noise_enabled=False, snr_threshold=0.)
                cloud = api.LidarSensor(cfg, api.SensorPose(**view['sensor_pose'])).scan(scene)
                self.assertEqual(cloud.n_beams, 1)
                self.assertEqual(cloud.n, 1)
                self.assertEqual(cloud.label_names[cloud.label_id[0]], 'study_target')
                self.assertAlmostEqual(cloud.range_true_m[0], view['range_m'])
                self.assertAlmostEqual(cloud.power_w[0]/setting['predicted_power_w'], 1., places=12)
                self.assertAlmostEqual(setting['predicted_power_w']/cfg.noise_rms_w, setting['requested_mean_snr'], places=11)
                self.assertGreaterEqual(setting['extinction_per_m'], 0.)
            with self.assertRaises(ValueError):
                prepare_design(temp, mean_snrs=(-1.,))

    def test_actual_scan_trial_files_keep_every_trial_and_unique_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'
            source.mkdir()
            self.source(source)
            destination = Path(temp)/'results'
            summary = run_study(destination, source, trials_per_setting=2)
            self.assertEqual(summary['aggregate']['trials'], 66)
            self.assertTrue((destination/'design.json').exists())
            seeds = []
            returned = 0
            for row in summary['settings']:
                with np.load(destination/row['trials_file']) as trials:
                    mask = trials['detected']
                    self.assertEqual(len(mask), 2)
                    self.assertEqual(int(mask.sum()), row['detections'])
                    self.assertTrue(np.isnan(trials['range_error_m'][~mask]).all())
                    self.assertTrue((trials['measured_snr'][mask] >= 10.).all())
                    seeds.extend(trials['seed'].tolist())
                    returned += int(mask.sum())
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertEqual(returned, summary['aggregate']['detections'])
            self.assertGreater(returned, 0)
            self.assertLess(returned, 66)


if __name__ == '__main__':
    unittest.main()
