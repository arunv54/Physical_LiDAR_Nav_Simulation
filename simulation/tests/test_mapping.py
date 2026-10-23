"""Measurement-only accumulation and temporal visibility checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lidar_project.mapping import build_map, cloud_at_step, load_map, observed_grid, save_map


def recorded(*scans, extent=(-2., 2., -2., 2.)):
    return dict(
        scans_available=True, scans=[{"xyz_world": np.asarray(xyz, float).reshape(-1, 3),
                                     "intensity": np.asarray(intensity, float)} for xyz, intensity in scans],
        controls={"t": np.arange(len(scans), dtype=float)},
        config={"planner": {"extent": list(extent)}, "robot": {"height_m": .1}},
        world_objects=object())  # Deliberately unusable: truth geometry is not a mapping input.


class MappingTests(unittest.TestCase):
    def test_voxel_representative_and_intensity_do_not_leak_future(self):
        source = recorded(([[.01, .01, .01]], [.1]),
                          ([[.09, .09, .09], [1., 0., .2]], [.9, .7]))
        mapping = build_map(source)
        self.assertEqual(mapping["metadata"]["map_point_count"], 2)
        first = cloud_at_step(mapping, 0)
        np.testing.assert_array_equal(first["xyz_world"], [[.01, .01, .01]])
        np.testing.assert_array_equal(first["intensity"], [.1])
        self.assertEqual(len(cloud_at_step(mapping, 1)["xyz_world"]), 2)
        self.assertEqual(len(cloud_at_step(mapping, -1)["xyz_world"]), 0)
        np.testing.assert_array_equal(cloud_at_step(mapping, 0)["xyz_world"], first["xyz_world"])
        self.assertEqual(len(cloud_at_step(mapping, 999)["xyz_world"]), 2)

    def test_negative_voxel_indices_use_floor(self):
        mapping = build_map(recorded(([[-.01, -.13, .2], [0., 0., 0.]], [.2, .3])))
        np.testing.assert_array_equal(mapping["map_voxel_index"], [[-1, -2, 1], [0, 0, 0]])

    def test_grid_precedence_and_backward_scrub(self):
        source = recorded(([[.25, .25, .01]], [.1]), ([[.25, .25, .2]], [.2]),
                          ([[.25, .25, 0.]], [.3]), extent=(0., 1., 0., 1.))
        mapping = build_map(source, voxel_size_m=.5)
        # Both heights share a 3D voxel: raster must still use all measurements.
        self.assertEqual(len(mapping["map_xyz"]), 1)
        self.assertEqual(observed_grid(mapping, 0)[1, 1], 1)
        self.assertEqual(observed_grid(mapping, 1)[1, 1], 2)
        self.assertEqual(observed_grid(mapping, 2)[1, 1], 2)
        self.assertEqual(observed_grid(mapping, 0)[1, 1], 1)
        self.assertEqual(mapping["ground_first_seen"][1, 1], 0)
        self.assertEqual(mapping["obstacle_first_seen"][1, 1], 1)
        self.assertEqual(np.count_nonzero(observed_grid(mapping)), 1)

    def test_crop_is_explicit_and_above_body_points_remain(self):
        source = recorded(([[-1., 0., 0.], [1., 0., 0.], [.3, .3, 5.], [np.nan, 0., 0.]],
                           [.1, .2, .3, .4]), extent=(-1., 1., -1., 1.))
        mapping = build_map(source)
        self.assertEqual(mapping["metadata"]["raw_return_count"], 4)
        self.assertEqual(mapping["metadata"]["finite_return_count"], 3)
        self.assertEqual(mapping["metadata"]["retained_measurement_count"], 2)
        self.assertEqual(mapping["metadata"]["map_z_range_m"][1], 5.)
        self.assertIsNone(mapping["metadata"]["z_limits_m"])
        cropped = build_map(source, z_limits_m=(-.5, 2.))
        self.assertEqual(cropped["metadata"]["retained_measurement_count"], 1)

    def test_no_rays_fill_unobserved_cells(self):
        mapping = build_map(recorded(([[1.1, 1.1, .2]], [.5])))
        self.assertEqual(np.count_nonzero(observed_grid(mapping)), 1)
        self.assertLess(mapping["metadata"]["observed_cell_fraction"], .01)

    def test_pickle_free_round_trip_preserves_temporal_views(self):
        mapping = build_map(recorded(([[.1, .1, 0.]], [.2]), ([[.1, .1, .4]], [.7])))
        with tempfile.TemporaryDirectory() as directory:
            save_map(mapping, directory)
            restored = load_map(directory)
            with np.load(Path(directory) / "mapping.npz", allow_pickle=False) as arrays:
                self.assertTrue(all(not arrays[key].dtype.hasobject for key in arrays.files))
            self.assertEqual(restored["metadata"], mapping["metadata"])
            for step in (-1, 0, 1, 10):
                np.testing.assert_array_equal(observed_grid(restored, step), observed_grid(mapping, step))
                np.testing.assert_array_equal(cloud_at_step(restored, step)["xyz_world"],
                                              cloud_at_step(mapping, step)["xyz_world"])

    def test_recorded_zero_scan_run_is_an_empty_map(self):
        mapping = build_map(recorded())
        self.assertEqual(mapping["map_xyz"].shape, (0, 3))
        self.assertEqual(np.count_nonzero(observed_grid(mapping)), 0)
        self.assertEqual(len(cloud_at_step(mapping, 0)["xyz_world"]), 0)

    def test_legacy_history_is_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "Full recorded 3D scans"):
            build_map({"scans_available": False, "scans": []})

    def test_non_aligned_crop_has_explicit_regular_raster_extent(self):
        source = recorded(([[1.05, .1, .2], [1.15, .1, .2]], [.2, .3]),
                          extent=(0., 1.1, 0., 1.))
        mapping = build_map(source)
        self.assertEqual(mapping["metadata"]["grid_shape"], [5, 6])
        self.assertEqual(mapping["metadata"]["roi_extent_m"], [0., 1.1, 0., 1.])
        self.assertAlmostEqual(mapping["metadata"]["extent_m"][1], 1.2)
        self.assertEqual(mapping["metadata"]["retained_measurement_count"], 1)


if __name__ == "__main__":
    unittest.main()
