"""Configuration loading and validation for the numerical model.

Legacy output switches are retained as validated configuration fields.  They
do not import a renderer or cause visualization while a scan is generated.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import math
import os
from dataclasses import replace
from typing import Optional, Sequence

from .model.geometry import Plane, Sphere, OrientedBox, VerticalCylinder, Scene
from .model.materials import Material, MATERIALS
from .model.sensor import LidarConfig, SensorPose


class ConfigError(ValueError):
    """Raised for any problem in the configuration file (message names the offending key)."""

def _fail(where: str, msg: str):
    raise ConfigError(f"{where}: {msg}")

def _strip_jsonc(text: str) -> str:
    """Blank out // and /* */ comments (outside strings) while preserving line/column positions."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)

def load_config_file(path: str) -> dict:
    """Read .json / .jsonc (comments allowed), .yaml / .yml (needs PyYAML) or .toml (Python >= 3.11)."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        if ext in (".json", ".jsonc"):
            data = json.loads(_strip_jsonc(text))
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigError("YAML config needs PyYAML (pip install pyyaml), or use a .jsonc file")
            data = yaml.safe_load(text)
        elif ext == ".toml":
            try:
                import tomllib
            except ImportError:
                raise ConfigError("TOML config needs Python >= 3.11, or use a .jsonc file")
            data = tomllib.loads(text)
        else:
            raise ConfigError(f"unsupported config extension '{ext}' (use .jsonc, .json, .yaml, .yml or .toml)")
    except (json.JSONDecodeError,) as e:
        raise ConfigError(f"{os.path.basename(path)}: syntax error at line {e.lineno}, column {e.colno}: {e.msg}")
    if not isinstance(data, dict):
        raise ConfigError(f"{os.path.basename(path)}: top level must be an object {{ ... }}")
    return data

def apply_overrides(raw: dict, items: Sequence[str]) -> dict:
    """--set section.key=value  (value parsed as JSON, else taken as a plain string)."""
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set '{item}': expected KEY=VALUE, e.g. sensor.rain_mm_per_h=25")
        key, val = item.split("=", 1)
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        parts, d = key.strip().split("."), raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ConfigError(f"--set '{item}': '{p}' is not a section")
        d[parts[-1]] = v
    return raw

def _check_keys(d, allowed, where):
    if not isinstance(d, dict):
        _fail(where, "must be an object { ... }")
    for k in d:
        if k in allowed or str(k).startswith("_"):
            continue
        hint = difflib.get_close_matches(str(k), sorted(allowed), n=1)
        _fail(where, f"unknown key '{k}'" + (f" - did you mean '{hint[0]}'?" if hint
                                             else f" (allowed: {', '.join(sorted(allowed))})"))

def _num(v, where, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        _fail(where, f"expected a number, got {v!r}")
    if lo is not None and (v < lo or (strict_lo and v == lo)):
        _fail(where, f"must be {'>' if strict_lo else '>='} {lo}, got {v}")
    if hi is not None and v > hi:
        _fail(where, f"must be <= {hi}, got {v}")
    return v

def _vec3(v, where, nonzero=False, positive=False):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        _fail(where, f"expected a list of 3 numbers [x, y, z], got {v!r}")
    out = [float(_num(c, f"{where}[{i}]", lo=0 if positive else None, strict_lo=positive)) for i, c in enumerate(v)]
    if nonzero and not any(out):
        _fail(where, "must not be the zero vector")
    return out

_SENSOR_BOUNDS = {   # name: (lo, hi, lo_is_exclusive)
    "n_channels": (1, None, False), "measurements_per_cycle": (1, None, False), "n_beam_samples": (1, None, False),
    "rotations_per_s": (0, None, True), "min_range_m": (0, None, False), "max_range_m": (0, None, True),
    "peak_power_w": (0, None, True), "beam_divergence_mrad": (0, None, False), "pulse_fwhm_ns": (0, None, True),
    "pulse_rise_ns": (0, None, True), "aperture_diam_m": (0, None, True), "optical_efficiency": (0, 1, True),
    "noise_rms_w": (0, None, True), "snr_threshold": (0, None, True), "saturation_power_w": (0, None, True),
    "range_floor_sigma_m": (0, None, False), "air_extinction_per_m": (0, None, False),
    "rain_mm_per_h": (0, None, False), "rain_a": (0, None, False), "rain_b": (0, None, False),
}

def build_sensor(sec: Optional[dict]) -> LidarConfig:
    sec = {} if sec is None else sec
    defaults = LidarConfig()
    _check_keys(sec, {f.name for f in dataclasses.fields(LidarConfig)}, "sensor")
    vals = {}
    for k, v in sec.items():
        if str(k).startswith("_"):
            continue
        w, dflt = f"sensor.{k}", getattr(defaults, k)
        if isinstance(dflt, bool):
            if not isinstance(v, bool):
                _fail(w, f"expected true/false, got {v!r}")
        elif isinstance(dflt, str):
            if not isinstance(v, str):
                _fail(w, f"expected a string, got {v!r}")
        else:
            lo, hi, strict = _SENSOR_BOUNDS.get(k, (None, None, False))
            v = _num(v, w, lo, hi, strict)
            if isinstance(dflt, int):
                if v != int(v):
                    _fail(w, f"expected an integer, got {v}")
                v = int(v)
            else:
                v = float(v)
        vals[k] = v
    cfg = replace(defaults, **vals)
    if cfg.return_mode not in ("first", "strongest", "last"):
        _fail("sensor.return_mode", f"must be 'first', 'strongest' or 'last', got {cfg.return_mode!r}")
    if cfg.vfov_upper_deg < cfg.vfov_lower_deg:
        _fail("sensor.vfov_upper_deg", "must be >= vfov_lower_deg")
    if cfg.hfov_end_deg <= cfg.hfov_start_deg:
        _fail("sensor.hfov_end_deg", "must be > hfov_start_deg")
    if cfg.min_range_m >= cfg.max_range_m:
        _fail("sensor.min_range_m", "must be < max_range_m")
    return cfg

def build_pose(sec: Optional[dict]) -> SensorPose:
    sec = {} if sec is None else sec
    _check_keys(sec, {f.name for f in dataclasses.fields(SensorPose)}, "pose")
    d = {}
    for k, v in sec.items():
        if str(k).startswith("_"):
            continue
        d[k] = _vec3(v, f"pose.{k}") if k in ("position", "velocity") else float(_num(v, f"pose.{k}"))
    return SensorPose(**d)

def _resolve_material(m, registry, where):
    if isinstance(m, str):
        if m not in registry:
            hint = difflib.get_close_matches(m, list(registry), n=1)
            _fail(where, f"unknown material '{m}'" + (f" - did you mean '{hint[0]}'?" if hint
                                                       else f" (known: {', '.join(registry)})"))
        return registry[m]
    if isinstance(m, dict):
        _check_keys(m, {"name", "rho_d", "rho_s", "spec_exp"}, where)
        if "rho_d" not in m:
            _fail(where, "inline material needs 'rho_d'")
        return Material(str(m.get("name", "inline")), float(_num(m["rho_d"], f"{where}.rho_d", 0)),
                        float(_num(m.get("rho_s", 0.0), f"{where}.rho_s", 0)),
                        float(_num(m.get("spec_exp", 50.0), f"{where}.spec_exp", 0, strict_lo=True)))
    _fail(where, f"expected a material name or an object, got {m!r}")

_OBJECT_KEYS = {  # type: (required, optional)
    "plane": ({"point", "normal"}, set()),
    "box": ({"center", "size"}, {"yaw_deg"}),
    "sphere": ({"center", "radius"}, set()),
    "cylinder": ({"base", "radius", "height"}, set()),
}

def build_scene(sec: Optional[dict]) -> Scene:
    if not isinstance(sec, dict):
        raise ConfigError("scene: missing - define at least one object under scene.objects")
    _check_keys(sec, {"materials", "objects"}, "scene")
    registry = dict(MATERIALS)
    for name, md in (sec.get("materials") or {}).items():
        registry[name] = _resolve_material({"name": name, **md} if isinstance(md, dict) else md, registry,
                                           f"scene.materials.{name}")
    objs = sec.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ConfigError("scene.objects: must be a non-empty list")
    prims, labels = [], set()
    for i, o in enumerate(objs):
        w = f"scene.objects[{i}]"
        if not isinstance(o, dict) or "type" not in o:
            _fail(w, f"needs a 'type' ({', '.join(_OBJECT_KEYS)})")
        typ = o["type"]
        if typ not in _OBJECT_KEYS:
            hint = difflib.get_close_matches(str(typ), list(_OBJECT_KEYS), n=1)
            _fail(w, f"unknown type {typ!r}" + (f" - did you mean '{hint[0]}'?" if hint else f" (use {', '.join(_OBJECT_KEYS)})"))
        req, opt = _OBJECT_KEYS[typ]
        _check_keys(o, req | opt | {"type", "material", "label", "enabled"}, w)
        for k in req:
            if k not in o:
                _fail(w, f"{typ} needs '{k}'")
        enabled = o.get("enabled", True)
        if not isinstance(enabled, bool):
            _fail(f"{w}.enabled", "expected true/false")
        label = str(o.get("label", f"{typ}_{i}"))
        if label in labels:
            _fail(f"{w}.label", f"duplicate label '{label}'")
        labels.add(label)
        if not enabled:
            continue
        mat = _resolve_material(o.get("material", "concrete"), registry, f"{w}.material")
        if typ == "plane":
            p = Plane(_vec3(o["point"], f"{w}.point"), _vec3(o["normal"], f"{w}.normal", nonzero=True), mat, label)
        elif typ == "box":
            p = OrientedBox(_vec3(o["center"], f"{w}.center"), _vec3(o["size"], f"{w}.size", positive=True),
                            float(_num(o.get("yaw_deg", 0.0), f"{w}.yaw_deg")), mat, label)
        elif typ == "sphere":
            p = Sphere(_vec3(o["center"], f"{w}.center"), _num(o["radius"], f"{w}.radius", 0, strict_lo=True), mat, label)
        else:
            p = VerticalCylinder(_vec3(o["base"], f"{w}.base"), _num(o["radius"], f"{w}.radius", 0, strict_lo=True),
                                 _num(o["height"], f"{w}.height", 0, strict_lo=True), mat, label)
        prims.append(p)
    if not prims:
        raise ConfigError("scene.objects: every object is disabled")
    return Scene(prims)

def build_all(raw: dict):
    """Validate a raw config dict -> (LidarConfig, SensorPose, Scene, simulation dict, output dict)."""
    _check_keys(raw, {"simulation", "sensor", "pose", "scene", "output"}, "config")
    sim = raw.get("simulation") or {}
    _check_keys(sim, {"seed"}, "simulation")
    seed = sim.get("seed", 1)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        _fail("simulation.seed", "expected an integer or null")
    out = {"directory": "../data/scan", "save": ["npz", "csv", "ply"], "static_png": False,
           "interactive_html": False, "open_browser": False, "title": None}
    user_out = raw.get("output") or {}
    _check_keys(user_out, set(out), "output")
    out.update({k: v for k, v in user_out.items() if not str(k).startswith("_")})
    if not isinstance(out["directory"], str) or not out["directory"]:
        _fail("output.directory", "expected a folder name")
    if not isinstance(out["save"], list) or not set(out["save"]) <= {"npz", "csv", "ply"}:
        _fail("output.save", "expected a list containing any of: npz, csv, ply")
    for k in ("static_png", "interactive_html", "open_browser"):
        if not isinstance(out[k], bool):
            _fail(f"output.{k}", "expected true/false")
    if out["title"] is not None and not isinstance(out["title"], str):
        _fail("output.title", "expected a string or null")
    return (build_sensor(raw.get("sensor")), build_pose(raw.get("pose")), build_scene(raw.get("scene")),
            {"seed": seed}, out)
