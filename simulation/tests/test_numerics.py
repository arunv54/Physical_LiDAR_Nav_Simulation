"""Seeded sensor references and numerical/presentation module boundaries."""
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lidar_project import api
from lidar_project.paths import PROJECT_ROOT

FIXTURES = Path(__file__).parent / "fixtures"


class SensorBehavior(unittest.TestCase):
    def test_original_seeded_scan_references(self):
        for path in sorted(FIXTURES.glob("sensor_*.npz")):
            with self.subTest(case=path.stem), np.load(path, allow_pickle=False) as reference:
                cfg, pose, scene, sim, _ = api.build_all(json.loads(str(reference["config_json"])))
                cloud = api.LidarSensor(cfg, pose, seed=sim["seed"]).scan(scene)
                for key in cloud.__dataclass_fields__:
                    actual, expected = np.asarray(getattr(cloud, key)), reference[key]
                    if expected.dtype.kind in "US":
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_saved_scan_round_trip(self):
        path = FIXTURES / "sensor_moving.npz"
        cloud = api.load_npz(path)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "scan.npz"
            cloud.save_npz(destination)
            loaded = api.load_npz(destination)
            for key in cloud.__dataclass_fields__:
                np.testing.assert_array_equal(np.asarray(getattr(loaded, key)), np.asarray(getattr(cloud, key)))


class ModuleBoundaries(unittest.TestCase):
    def test_numerical_pipeline_needs_no_visualization(self):
        # Refuse optional presentation imports even if installed on this machine.
        script = '''
import importlib.abc, sys
class BlockVisuals(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib','plotly','PIL','tkinter'} or fullname.startswith('lidar_project.visualization'):
            raise RuntimeError('Numerical layer imported visualization: '+fullname)
sys.meta_path.insert(0, BlockVisuals())
from lidar_project.cli import main
from lidar_project import api
from lidar_project.trajectory import load_config, run_trajectory, save_result, load_result
config = load_config()
config['robot']['max_steps'] = 2
config['sensor_overrides'].update(n_channels=2, measurements_per_cycle=24)
save_result(run_trajectory(config, seed=1), sys.argv[1])
from lidar_project.mapping import build_map
mapped = build_map(load_result(sys.argv[1]))
assert mapped['metadata']['steps'] == 2
'''
        with tempfile.TemporaryDirectory() as temporary:
            finished = subprocess.run([sys.executable, "-c", script, str(Path(temporary)/"trajectory")],
                                      cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(finished.returncode, 0, finished.stdout + finished.stderr)

    def test_config_resolution_does_not_depend_on_working_directory(self):
        from lidar_project.trajectory import load_config
        config = load_config()
        self.assertTrue(Path(config['_source']).is_absolute())
        self.assertIn('n_channels', config['_sensor_base_snapshot'])


if __name__ == "__main__":
    unittest.main()
