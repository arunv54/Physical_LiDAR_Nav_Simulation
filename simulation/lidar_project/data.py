"""Point-cloud arrays and lossless scan storage; no rendering dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PointCloud:
    xyz_sensor: np.ndarray          # (M,3) raw measurement r*dir in the sensor frame (shows motion skew)
    xyz_world: np.ndarray           # (M,3) positions using the true sensor pose at firing time
    range_m: np.ndarray             # measured (noisy) range
    range_true_m: np.ndarray        # noise-free echo range
    intensity: np.ndarray           # 0..1, echo power / saturation power
    reflectivity: np.ndarray        # apparent range-compensated return; not intrinsic reflectance
    power_w: np.ndarray             # measured optical echo power
    snr: np.ndarray
    incidence_deg: np.ndarray
    ring: np.ndarray                # channel index
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    time_s: np.ndarray              # firing time within the sweep
    label_id: np.ndarray            # ground truth object index
    label_names: List[str]
    range_image: np.ndarray         # (n_channels, n_azimuth), NaN = no return
    n_beams: int

    @property
    def n(self) -> int:
        return len(self.range_m)

    def label_name(self) -> np.ndarray:
        return np.array(self.label_names)[self.label_id]

    def save_npz(self, path: str):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "label_names"}
        np.savez_compressed(path, label_names=np.array(self.label_names), **d)

    def _table(self):
        cols = [self.xyz_sensor[:, 0], self.xyz_sensor[:, 1], self.xyz_sensor[:, 2],
                self.intensity, self.reflectivity, self.range_m, self.snr,
                self.ring, self.azimuth_deg, self.time_s, self.label_id]
        names = "x y z intensity reflectivity range snr ring azimuth_deg time_s label_id".split()
        return np.column_stack(cols), names

    def save_csv(self, path: str):
        tab, names = self._table()
        np.savetxt(path, tab, delimiter=",", header=",".join(names), comments="",
                   fmt=["%.4f"] * 3 + ["%.5f", "%.5f", "%.4f", "%.2f", "%d", "%.3f", "%.6f", "%d"])

    def save_ply(self, path: str):
        """ASCII PLY (opens in CloudCompare / MeshLab / Open3D) in the sensor frame."""
        tab, _ = self._table()
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\ncomment sensor frame (x fwd, y left, z up)\n")
            f.write(f"element vertex {self.n}\n")
            for p in ("x", "y", "z", "intensity", "reflectivity"):
                f.write(f"property float {p}\n")
            f.write("property int label\nend_header\n")
            np.savetxt(f, np.column_stack([tab[:, :3], tab[:, 3], tab[:, 4], tab[:, 10]]),
                       fmt="%.4f %.4f %.4f %.5f %.5f %d")

    @classmethod
    def load_npz(cls, path: str) -> "PointCloud":
        """Read the numerical arrays saved by :meth:`save_npz`."""
        return load_npz(path)


def load_npz(path: str) -> PointCloud:
    """Load an existing scan without sensor simulation or plotting imports.

    NPZ files are read with pickle disabled.  Every current PointCloud field is
    required, preventing silent reconstruction of missing measurement metadata.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(PointCloud.__dataclass_fields__) - set(archive.files))
        if missing:
            raise ValueError("scan archive is missing fields: " + ", ".join(missing))
        values = {name: np.array(archive[name], copy=True)
                  for name in PointCloud.__dataclass_fields__}
    values["label_names"] = [str(name) for name in values["label_names"].tolist()]
    values["n_beams"] = int(values["n_beams"].item())
    return PointCloud(**values)
