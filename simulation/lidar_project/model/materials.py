"""Illustrative diffuse/specular optical return coefficients."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Illustrative NIR return coefficients; retroreflectors use an effective gain."""
    name: str
    rho_d: float                 # diffuse (Lambertian) reflectance, 0..1 (retro-reflectors may exceed 1)
    rho_s: float = 0.0           # weight of the specular lobe
    spec_exp: float = 50.0       # sharpness of the specular lobe


MATERIALS = {m.name: m for m in [
    Material("asphalt", 0.10),
    Material("concrete", 0.30),
    Material("brick", 0.25),
    Material("wood", 0.45),
    Material("aluminium", 0.55, rho_s=0.20, spec_exp=60.0),
    Material("white_paint", 0.85),
    Material("car_paint_black", 0.05, rho_s=0.30, spec_exp=120.0),
    Material("retroreflector", 4.0),
]}
