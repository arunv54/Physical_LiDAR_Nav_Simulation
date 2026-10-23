"""Video chronology and controlled-comparison invariants."""
import copy
import unittest

import numpy as np

from lidar_project.visualization.simulation_video import RecordedRun, _same_scene


class SimulationVideoTests(unittest.TestCase):
    def test_common_clock_freezes_terminated_run_without_future_scan_or_voxels(self):
        run=RecordedRun("test",{"status":"collision"},{
            "states_t":np.array([0.,.2,.4]),"controls_t":np.array([0.,.2]),
            "scan_offsets":np.array([0,1,2]),"scan_xyz_world":np.array([[0,0,0],[1,0,0]]),
            "scan_target_mask":np.array([True,False])},
            {"map_first_seen":np.array([0,1]),"ground_first_seen":np.array([[0,-1]]),
             "obstacle_first_seen":np.array([[-1,1]])},{},(),
             {"detected_returns":np.array([1.,0.])})
        early=run.at(.1)
        self.assertEqual(early["scan"],0)
        self.assertEqual(early["state"],0)
        self.assertEqual(early["map_voxels"],1)
        self.assertEqual(early["observed_fraction"],.5)
        self.assertFalse(early["ended"])
        self.assertEqual(len(early["target_xyz"]),1)
        self.assertEqual(early["diagnostics"]["detected_returns"],1)
        terminal=run.at(.4); held=run.at(20.)
        self.assertTrue(held["ended"])
        self.assertEqual(held["state"],2)
        self.assertEqual(held["scan"],1)
        self.assertEqual(held["map_voxels"],2)
        np.testing.assert_array_equal(terminal["xyz"],held["xyz"])
        self.assertEqual(len(held["target_xyz"]),0)
        self.assertEqual(held["diagnostics"]["detected_returns"],0)

    def controlled_run(self):
        return RecordedRun("dense_bright",{
            "world_objects":[{"type":"box","label":"other","center":[1,0,1],"material":{"rho_d":.3}},
                             {"type":"box","label":"study_target","center":[11,1.1,.6],"material":{"rho_d":.85,"rho_s":0}}],
            "sensor":{"measurements_per_cycle":360,"hfov_start_deg":-180,"hfov_end_deg":180,"noise_rms_w":4e-9},
            "config":{"mount_height_m":.6,"sensor_mount":{"pitch_deg":0,"yaw_deg":0,"roll_deg":0}}},
            {"reference_xy":np.array([[0.,0.],[18.,0.]]),"states_t":np.array([0.,.2]),
             "states_xy":np.array([[0.,0.],[.1,0.]]),"states_theta":np.zeros(2),"controls_t":np.array([0.])},
            {},{},())

    def test_factorial_allows_only_sampling_and_designated_diffuse_reflectance(self):
        dense=self.controlled_run();sparse=copy.deepcopy(dense)
        sparse.metadata["sensor"]["measurements_per_cycle"]=90
        sparse.metadata["world_objects"][1]["material"]["rho_d"]=.00005
        _same_scene([dense,sparse],controlled=True)
        with self.assertRaisesRegex(ValueError,"geometry/material"):
            _same_scene([dense,sparse])
        for mutation in ("angle","noise","other_material","target_geometry","pose"):
            changed=copy.deepcopy(sparse)
            if mutation=="angle":
                changed.metadata["config"]["sensor_mount"]["pitch_deg"]=15
            elif mutation=="noise":
                changed.metadata["sensor"]["noise_rms_w"]=4e-8
            elif mutation=="other_material":
                changed.metadata["world_objects"][0]["material"]["rho_d"]=.01
            elif mutation=="target_geometry":
                changed.metadata["world_objects"][1]["center"][0]=12
            else:
                changed.arrays["states_theta"][0]=.2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _same_scene([dense,changed],controlled=True)


if __name__=="__main__":
    unittest.main()
