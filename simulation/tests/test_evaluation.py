"""Check units, boundary cases and conventions used by the analytical oracles."""
import math
import unittest

import numpy as np

from lidar_project import api
from lidar_project.experiments.geometry import nominal_fov_mask
from lidar_project.experiments.response import reference_power, wilson_interval


class EvaluationReferences(unittest.TestCase):
    def test_diffuse_reference_units_and_incidence(self):
        cfg = api.LidarConfig(air_extinction_per_m=0, rain_mm_per_h=0)
        # 75 W, 25 mm aperture and efficiency 0.5 at 10 m give this
        # independently reduced on-axis diffuse power for input weight 0.30.
        expected = 1.7578125e-5
        self.assertAlmostEqual(reference_power(cfg, .30, 10), expected, places=16)
        self.assertAlmostEqual(reference_power(cfg, .30, 20), expected / 4, places=16)
        self.assertAlmostEqual(reference_power(cfg, .30, 10, 60), expected / 2, places=16)

    def test_nominal_fov_wrapping_and_inclusive_boundaries(self):
        cfg = api.LidarConfig(hfov_start_deg=170, hfov_end_deg=190,
                              vfov_lower_deg=-5, vfov_upper_deg=5)
        angles = np.radians([169, 170, 180, -170, -169, 0])
        points = np.column_stack((10*np.cos(angles), 10*np.sin(angles), np.zeros(len(angles))))
        np.testing.assert_array_equal(nominal_fov_mask(points, cfg, api.SensorPose()),
                                      [False, True, True, True, False, False])
        cfg = api.replace(cfg, hfov_start_deg=-2, hfov_end_deg=2,
                          vfov_lower_deg=-1, vfov_upper_deg=1)
        down = [[10*math.cos(math.radians(10)), 0, -10*math.sin(math.radians(10))]]
        self.assertTrue(nominal_fov_mask(down, cfg, api.SensorPose(pitch_deg=10))[0])
        self.assertFalse(nominal_fov_mask(down, cfg, api.SensorPose())[0])

    def test_binomial_interval_keeps_endpoint_cases(self):
        low, high = wilson_interval(0, 1000)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        low_all, high_all = wilson_interval(1000, 1000)
        self.assertEqual(high_all, 1)
        self.assertAlmostEqual(1-high, low_all)
        middle = wilson_interval(500, 1000)
        self.assertLess(middle[0], .5)
        self.assertGreater(middle[1], .5)


if __name__ == '__main__':
    unittest.main()
