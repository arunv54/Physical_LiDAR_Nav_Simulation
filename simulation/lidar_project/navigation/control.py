"""Point-based steering geometry used by the report's trajectory controller."""
from __future__ import annotations

import numpy as np


def wrap(a):
    """Wrap angles in radians to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def free_path_lengths(pts, xy, headings, r_inf, lookahead):
    """Distance a disc can translate along each heading before touching a point."""
    L = np.full(len(headings), float(lookahead))
    if len(pts) == 0:
        return L
    q = pts - xy
    u = np.stack([np.cos(headings), np.sin(headings)], 1)
    along = q @ u.T
    perp = np.abs(q[:, 0, None] * u[None, :, 1] - q[:, 1, None] * u[None, :, 0])
    hit = (perp < r_inf) & (along > -r_inf)
    contact = along - np.sqrt(np.maximum(r_inf ** 2 - perp ** 2, 0.0))
    contact = np.where(hit, contact, np.inf)
    return np.clip(contact.min(axis=0), 0.0, lookahead)
