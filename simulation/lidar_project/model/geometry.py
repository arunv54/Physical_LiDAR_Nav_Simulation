"""Numerical ray intersections and coordinate transforms, without display meshes."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .materials import MATERIALS, Material

EPS = 1e-6


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

def flu_to_ned(xyz: np.ndarray) -> np.ndarray:
    """Forward-Left-Up -> North-East-Down style axes (x, -y, -z)."""
    return xyz * np.array([1.0, -1.0, -1.0])

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)

def _face_ray(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Flip normals so that they point back toward the ray origin (n . d <= 0)."""
    s = np.sign(_dot(n, d))
    s[s == 0] = 1.0
    return n * (-s)[:, None]

class Primitive:
    material: Material
    label: str

    def intersect(self, o: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class Plane(Primitive):
    """Infinite plane through `point` with normal `normal` (ground, wall...)."""

    def __init__(self, point, normal, material="concrete", label="plane"):
        self.p = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        denom = d @ self.n
        num = (self.p - o) @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / denom
        ok = (np.abs(denom) > 1e-9) & (t > EPS)
        t = np.where(ok, t, np.inf)
        n = np.tile(self.n, (len(t), 1))
        return t, _face_ray(n, d)

class Sphere(Primitive):
    def __init__(self, center, radius, material="aluminium", label="sphere"):
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        oc = o - self.c
        b = _dot(oc, d)
        cc = _dot(oc, oc) - self.r ** 2
        disc = b * b - cc
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1, t2 = -b - sq, -b + sq
        t = np.where(t1 > EPS, t1, t2)                 # t2 only if the origin is inside
        t = np.where(ok & (t > EPS), t, np.inf)
        p = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        return t, _face_ray((p - self.c) / self.r, d)

class OrientedBox(Primitive):
    """Box with edge lengths `size` (x, y, z in the box frame), rotated by `yaw_deg` about z."""

    def __init__(self, center, size, yaw_deg=0.0, material="concrete", label="box"):
        self.c = np.asarray(center, float)
        self.h = 0.5 * np.asarray(size, float)
        self.R = rot_z(np.radians(yaw_deg))
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        lo = (o - self.c) @ self.R                      # = R^T (o - c), row-wise
        ld = d @ self.R
        ld = np.where(np.abs(ld) < 1e-12, 1e-12, ld)
        inv = 1.0 / ld
        t0, t1 = (-self.h - lo) * inv, (self.h - lo) * inv
        tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
        t_near, t_far = tmin.max(axis=1), tmax.min(axis=1)
        ok = (t_near <= t_far) & (t_far > EPS)
        inside = t_near <= EPS
        t = np.where(ok, np.where(inside, t_far, t_near), np.inf)
        ax = np.where(inside, tmax.argmin(axis=1), tmin.argmax(axis=1))
        idx = np.arange(len(lo))
        nl = np.zeros_like(lo)
        nl[idx, ax] = -np.sign(ld[idx, ax])
        return t, nl @ self.R.T

class VerticalCylinder(Primitive):
    """Finite cylinder with its axis along +z, base centre `base`, capped (poles, trunks, tanks)."""

    def __init__(self, base, radius, height, material="wood", label="cylinder"):
        self.base = np.asarray(base, float)
        self.r = float(radius)
        self.hgt = float(height)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        N = len(o)
        rel = o - self.base
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        ox, oy, oz = rel[:, 0], rel[:, 1], rel[:, 2]
        a = dx * dx + dy * dy
        b = ox * dx + oy * dy
        c = ox * ox + oy * oy - self.r ** 2
        disc = b * b - a * c
        a_ok = (a > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.where(a_ok, disc, 0.0))
        a_safe = np.where(a > 1e-12, a, 1.0)
        cand = []
        for sgn in (-1.0, 1.0):                         # side wall, near/far root
            t = (-b + sgn * sq) / a_safe
            z = oz + t * dz
            cand.append(np.where(a_ok & (t > EPS) & (z >= 0) & (z <= self.hgt), t, np.inf))
        dz_ok = np.abs(dz) > 1e-12
        dz_safe = np.where(dz_ok, dz, 1.0)
        for zc in (0.0, self.hgt):                      # bottom / top cap
            t = (zc - oz) / dz_safe
            x, y = ox + t * dx, oy + t * dy
            cand.append(np.where(dz_ok & (t > EPS) & (x * x + y * y <= self.r ** 2), t, np.inf))
        T = np.stack(cand)                               # (4, N)
        k = T.argmin(axis=0)
        t = T[k, np.arange(N)]
        p = rel + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        n = np.zeros((N, 3))
        side = k < 2
        n[side, 0] = p[side, 0] / self.r
        n[side, 1] = p[side, 1] / self.r
        n[k == 2, 2] = -1.0
        n[k == 3, 2] = 1.0
        return t, _face_ray(n, d)

class Scene:
    def __init__(self, objects: Sequence[Primitive]):
        self.objects = list(objects)
        self.labels = [o.label for o in self.objects]
        self.rho_d = np.array([o.material.rho_d for o in self.objects] + [0.0])
        self.rho_s = np.array([o.material.rho_s for o in self.objects] + [0.0])
        self.spec_exp = np.array([o.material.spec_exp for o in self.objects] + [1.0])

    def cast(self, o: np.ndarray, d: np.ndarray):
        """Closest hit per ray -> (t, object index (-1 = miss), normal)."""
        N = len(d)
        best_t = np.full(N, np.inf)
        best_i = np.full(N, -1, dtype=np.int32)
        best_n = np.zeros((N, 3))
        for i, obj in enumerate(self.objects):
            t, n = obj.intersect(o, d)
            m = t < best_t
            best_t[m], best_i[m], best_n[m] = t[m], i, n[m]
        return best_t, best_i, best_n
