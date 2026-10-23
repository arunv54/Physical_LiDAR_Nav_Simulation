"""Spinning pulsed LiDAR forward model and assumed receiver uncertainty.

The scanner uses nearest opaque intersections, Gaussian-weighted sub-rays,
simplified diffuse/specular optical power, echo centroids, threshold detection,
and a timing-inspired conditional range-noise law.  It does not sample a
received waveform or claim a calibrated reflectivity estimate or timing bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..data import PointCloud
from .geometry import Scene, _dot, rot_x, rot_y

C_LIGHT = 299_792_458.0


@dataclass
class LidarConfig:
    # --- scan geometry (Cosys: NumberOfChannels, MeasurementsPerCycle, RotationsPerSecond, FOV...)
    n_channels: int = 32
    vfov_lower_deg: float = -25.0
    vfov_upper_deg: float = 15.0
    measurements_per_cycle: int = 1800           # azimuth steps per 360 deg
    hfov_start_deg: float = -180.0
    hfov_end_deg: float = 180.0
    rotations_per_s: float = 10.0
    min_range_m: float = 0.5
    max_range_m: float = 200.0                   # hard time-of-flight gate (Cosys: Range)
    # --- transmitter
    peak_power_w: float = 75.0
    beam_divergence_mrad: float = 2.0            # full angle, 1/e^2 intensity
    n_beam_samples: int = 12                     # sub-rays per pulse (1 = pencil beam)
    pulse_fwhm_ns: float = 5.0                   # echoes closer than c*tau/2 merge
    pulse_rise_ns: float = 1.0
    # --- receiver
    aperture_diam_m: float = 0.025
    optical_efficiency: float = 0.5
    noise_rms_w: float = 4.0e-9                  # receiver noise (rms, in optical-power units)
    snr_threshold: float = 10.0
    saturation_power_w: float = 5.0e-5           # power mapped to intensity = 1
    range_floor_sigma_m: float = 0.01
    noise_enabled: bool = True
    # --- atmosphere (rain constants = Cosys-AirSim defaults rainConstantA / rainConstantB)
    air_extinction_per_m: float = 2.0e-4         # ~20 km visibility
    rain_mm_per_h: float = 0.0
    rain_a: float = 0.01
    rain_b: float = 0.6
    # --- echo selection
    return_mode: str = "strongest"               # 'first' | 'strongest' | 'last'

    # derived quantities ------------------------------------------------------
    @property
    def rx_area_m2(self) -> float:
        return np.pi * (0.5 * self.aperture_diam_m) ** 2

    @property
    def extinction_per_m(self) -> float:
        rain = self.rain_a * self.rain_mm_per_h ** self.rain_b if self.rain_mm_per_h > 0 else 0.0
        return self.air_extinction_per_m + rain

    @property
    def link_const(self) -> float:               # P_t * A_r * eta
        return self.peak_power_w * self.rx_area_m2 * self.optical_efficiency

    @property
    def scan_period_s(self) -> float:
        return 1.0 / self.rotations_per_s

    def elevations_deg(self) -> np.ndarray:
        if self.n_channels == 1:                 # Cosys convention: 2D lidar uses the upper angle
            return np.array([self.vfov_upper_deg])
        return np.linspace(self.vfov_lower_deg, self.vfov_upper_deg, self.n_channels)

    def azimuths_deg(self) -> np.ndarray:
        step = 360.0 / self.measurements_per_cycle
        span = self.hfov_end_deg - self.hfov_start_deg
        n = self.measurements_per_cycle if span >= 360.0 - 1e-9 else int(np.floor(span / step + 1e-9)) + 1
        return self.hfov_start_deg + step * np.arange(n)

def received_power(cfg: LidarConfig, rho: float, cos_inc: float, R: float) -> float:
    """LiDAR equation for a Lambertian target that fills the beam."""
    return (cfg.link_const * rho / np.pi * cos_inc / R ** 2) * np.exp(-2.0 * cfg.extinction_per_m * R)

def max_range_estimate(cfg: LidarConfig, rho: float = 0.1, cos_inc: float = 1.0) -> float:
    """Range where the mean echo power equals the detection threshold (50 % detection)."""
    p_min = cfg.snr_threshold * cfg.noise_rms_w
    lo, hi = 0.05, 1.0e5
    for _ in range(200):                          # power decreases monotonically with R
        mid = np.sqrt(lo * hi)
        if received_power(cfg, rho, cos_inc, mid) > p_min:
            lo = mid
        else:
            hi = mid
    return float(lo)

@dataclass
class SensorPose:
    position: Sequence[float] = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    velocity: Sequence[float] = (0.0, 0.0, 0.0)          # world m/s, applied during the sweep
    yaw_rate_deg_s: float = 0.0

def beam_pattern(n: int, divergence_mrad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-area (sunflower) sampling of a Gaussian beam: offsets (rad) and energy weights."""
    if n <= 1:
        return np.zeros((1, 2)), np.ones(1)
    theta0 = 0.5 * divergence_mrad * 1e-3            # 1/e^2 half-angle
    sigma = theta0 / 2.0                              # I ~ exp(-2 r^2/theta0^2) = exp(-r^2/(2 sigma^2))
    r_max = 2.5 * sigma                               # contains 95.6 % of the energy
    i = np.arange(n)
    r = r_max * np.sqrt((i + 0.5) / n)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    w = np.exp(-r ** 2 / (2.0 * sigma ** 2))
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1), w / w.sum()

class LidarSensor:
    def __init__(self, cfg: LidarConfig, pose: Optional[SensorPose] = None, seed: Optional[int] = 0):
        self.cfg = cfg
        self.pose = pose or SensorPose()
        self.rng = np.random.default_rng(seed)

    # ---- one full sweep ------------------------------------------------------
    def scan(self, scene: Scene) -> PointCloud:
        cfg, pose = self.cfg, self.pose
        el = np.radians(cfg.elevations_deg())
        az_deg = cfg.azimuths_deg()
        az = np.radians(az_deg)
        n_az, n_ch = len(az), len(el)
        AZ, EL = np.meshgrid(az, el, indexing="ij")                      # (n_az, n_ch)
        ring = np.tile(np.arange(n_ch), (n_az, 1)).ravel()
        az_deg_flat = np.degrees(AZ).ravel()
        el_deg_flat = np.degrees(EL).ravel()
        N = n_az * n_ch

        # beam direction in the sensor frame
        d_loc = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)

        # firing time of every pulse, and sensor pose at that time (rolling-scan geometry)
        t_fire = ((np.degrees(AZ) - cfg.hfov_start_deg) / 360.0 * cfg.scan_period_s).ravel()
        origin = np.asarray(pose.position, float) + np.asarray(pose.velocity, float) * t_fire[:, None]
        yaw_t = np.radians(pose.yaw_deg + pose.yaw_rate_deg_s * t_fire)
        R_mount = rot_y(np.radians(pose.pitch_deg)) @ rot_x(np.radians(pose.roll_deg))
        v = d_loc @ R_mount.T
        c, s = np.cos(yaw_t), np.sin(yaw_t)
        d_world = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)

        # ---- beam sub-rays (Gaussian profile) ----
        K = max(1, cfg.n_beam_samples)
        off, w = beam_pattern(K, cfg.beam_divergence_mrad)
        helper = np.tile([0.0, 0.0, 1.0], (N, 1))
        helper[np.abs(d_world[:, 2]) > 0.99] = [1.0, 0.0, 0.0]
        u = np.cross(d_world, helper)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        vv = np.cross(d_world, u)
        sub = d_world[:, None, :] + off[None, :, 0, None] * u[:, None, :] + off[None, :, 1, None] * vv[:, None, :]
        sub /= np.linalg.norm(sub, axis=2, keepdims=True)

        # ---- ray casting ----
        o_rep = np.repeat(origin, K, axis=0)
        d_flat = sub.reshape(-1, 3)
        t, idx, nrm = scene.cast(o_rep, d_flat)
        valid = (idx >= 0) & (t >= cfg.min_range_m) & (t <= cfg.max_range_m)

        # ---- radiometry: LiDAR equation with Lambertian + specular BRDF ----
        cos_t = np.clip(-_dot(nrm, d_flat), 0.0, 1.0)
        i_safe = np.where(idx >= 0, idx, len(scene.objects))
        cos2 = np.clip(2.0 * cos_t ** 2 - 1.0, 0.0, 1.0)                 # monostatic: lobe angle = 2*theta
        f_r = (scene.rho_d[i_safe] / np.pi
               + scene.rho_s[i_safe] * (scene.spec_exp[i_safe] + 2.0) / (2.0 * np.pi) * cos2 ** scene.spec_exp[i_safe])
        Rr = np.where(valid, t, 1.0)
        P = cfg.link_const * f_r * cos_t / Rr ** 2 * np.exp(-2.0 * cfg.extinction_per_m * Rr)
        P = np.where(valid, P, 0.0).reshape(N, K) * w[None, :]
        R = np.where(valid, t, np.inf).reshape(N, K)
        ID = idx.reshape(N, K)
        COS = cos_t.reshape(N, K)

        found, r_true, p_true, p_meas, lab, inc = self._detect(R, P, ID, COS)

        # ---- range noise from the echo SNR ----
        snr = p_meas / cfg.noise_rms_w
        r_meas = r_true.copy()
        if cfg.noise_enabled:
            sig = np.sqrt(cfg.range_floor_sigma_m ** 2
                          + (0.5 * C_LIGHT * cfg.pulse_rise_ns * 1e-9) ** 2 / (2.0 * np.maximum(snr, 1e-9)))
            r_meas = r_true + self.rng.normal(0.0, 1.0, N) * sig

        f = found
        pts_sensor = r_meas[f, None] * d_loc[f]
        pts_world = origin[f] + r_meas[f, None] * d_world[f]
        rho_app = p_meas[f] * np.pi * r_meas[f] ** 2 / cfg.link_const
        rimg = np.full(N, np.nan)
        rimg[f] = r_meas[f]
        return PointCloud(
            xyz_sensor=pts_sensor, xyz_world=pts_world, range_m=r_meas[f], range_true_m=r_true[f],
            intensity=np.clip(p_meas[f] / cfg.saturation_power_w, 0.0, 1.0), reflectivity=rho_app,
            power_w=p_meas[f], snr=snr[f], incidence_deg=np.degrees(np.arccos(np.clip(inc[f], 0, 1))),
            ring=ring[f], azimuth_deg=az_deg_flat[f], elevation_deg=el_deg_flat[f], time_s=t_fire[f],
            label_id=lab[f], label_names=list(scene.labels),
            range_image=rimg.reshape(n_az, n_ch).T, n_beams=N)

    # ---- echo formation: merge sub-ray returns inside the pulse gate, then detect ----
    def _detect(self, R, P, ID, COS):
        cfg, rng = self.cfg, self.rng
        N, K = R.shape
        order = np.argsort(R, axis=1)
        R = np.take_along_axis(R, order, 1)
        P = np.take_along_axis(P, order, 1)
        ID = np.take_along_axis(ID, order, 1)
        COS = np.take_along_axis(COS, order, 1)
        gate = 0.5 * C_LIGHT * cfg.pulse_fwhm_ns * 1e-9

        opened = np.zeros(N, bool)
        cP, cPR, cLast = np.zeros(N), np.zeros(N), np.zeros(N)
        cBestP, cBestID, cBestCos = np.zeros(N), np.full(N, -1, np.int32), np.zeros(N)
        found = np.zeros(N, bool)
        out_r, out_pt, out_pm = np.zeros(N), np.zeros(N), np.zeros(N)
        out_id, out_cos = np.full(N, -1, np.int32), np.zeros(N)

        def finalize(mask):
            sel = np.flatnonzero(mask)
            if sel.size == 0:
                return
            pt = cP[sel]
            pm = pt + (rng.normal(0.0, cfg.noise_rms_w, sel.size) if cfg.noise_enabled else 0.0)
            det = pm / cfg.noise_rms_w >= cfg.snr_threshold
            rc = cPR[sel] / np.maximum(pt, 1e-300)
            if cfg.return_mode == "first":
                take = det & ~found[sel]
            elif cfg.return_mode == "last":
                take = det
            else:
                take = det & (~found[sel] | (pm > out_pm[sel]))
            s2 = sel[take]
            out_r[s2], out_pt[s2], out_pm[s2] = rc[take], pt[take], pm[take]
            out_id[s2], out_cos[s2] = cBestID[s2], cBestCos[s2]
            found[s2] = True

        for k in range(K):
            r, p = R[:, k], P[:, k]
            hit = np.isfinite(r)
            new = hit & (~opened | ((r - cLast) > gate))
            finalize(new & opened)
            cP[new], cPR[new], cLast[new] = p[new], p[new] * r[new], r[new]
            cBestP[new], cBestID[new], cBestCos[new] = p[new], ID[new, k], COS[new, k]
            opened[new] = True
            add = hit & ~new
            cP[add] += p[add]
            cPR[add] += p[add] * r[add]
            cLast[add] = r[add]
            better = add & (p > cBestP)
            cBestP[better], cBestID[better], cBestCos[better] = p[better], ID[better, k], COS[better, k]
        finalize(opened)
        return found, out_r, out_pt, out_pm, out_id, out_cos
