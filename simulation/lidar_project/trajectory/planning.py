"""Known-map route generation and exact swept-disc scoring primitives."""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .. import api


def _point_segment(point, a, b):
    delta = b - a
    u = float(np.dot(point - a, delta) / np.dot(delta, delta)) if np.dot(delta, delta) else 0.0
    return float(np.linalg.norm(point - (a + np.clip(u, 0, 1) * delta)))


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_distance(a, b, c, d):
    ab, cd = b - a, d - c
    denominator = _cross(ab, cd)
    if abs(denominator) > 1e-14:
        u, v = _cross(c - a, cd) / denominator, _cross(c - a, ab) / denominator
        if -1e-12 <= u <= 1 + 1e-12 and -1e-12 <= v <= 1 + 1e-12:
            return 0.0
    return min(_point_segment(a, c, d), _point_segment(b, c, d),
               _point_segment(c, a, b), _point_segment(d, a, b))


def _footprint_segment_distance(a, b, obj, height):
    if isinstance(obj, api.Plane):
        if abs(obj.n[2]) < 0.99:
            raise ValueError("Trajectory scoring requires finite wall boxes, not vertical infinite planes")
        return math.inf
    if isinstance(obj, api.OrientedBox):
        if obj.c[2] - obj.h[2] > height or obj.c[2] + obj.h[2] < 0:
            return math.inf
        aa, bb = (a - obj.c[:2]) @ obj.R[:2, :2], (b - obj.c[:2]) @ obj.R[:2, :2]
        if np.all(np.abs(aa) <= obj.h[:2]) or np.all(np.abs(bb) <= obj.h[:2]):
            return 0.0
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * obj.h[:2]
        return min(_segments_distance(aa, bb, corners[i], corners[(i + 1) % 4]) for i in range(4))
    if isinstance(obj, api.VerticalCylinder):
        if obj.base[2] > height or obj.base[2] + obj.hgt < 0:
            return math.inf
        return max(0.0, _point_segment(obj.base[:2], a, b) - obj.r)
    if isinstance(obj, api.Sphere):
        dz = max(-obj.c[2], obj.c[2] - height, 0.0)
        if dz > obj.r:
            return math.inf
        radius = math.sqrt(max(obj.r ** 2 - dz ** 2, 0))
        return max(0.0, _point_segment(obj.c[:2], a, b) - radius)
    raise TypeError(f"Unsupported trajectory obstacle {type(obj).__name__}")


def swept_surface_gap(a, b, scene_or_objects, radius, height):
    """Minimum path-to-filled-footprint distance minus disc radius, in metres.

    This exactly scores the straight segment used by forward Euler, including
    between-sample collisions. A negative value means overlap; values inside a
    filled obstacle saturate at -radius, rather than describing penetration depth.
    Only obstacles overlapping the robot's vertical interval [0,height] count.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    objects = getattr(scene_or_objects, "objects", scene_or_objects)
    if objects and isinstance(objects[0], dict):
        objects = api.build_scene({"objects": objects}).objects
    return min((_footprint_segment_distance(a, b, obj, height) - radius for obj in objects), default=math.inf)


def resample_route(polyline, spacing):
    polyline = np.asarray(polyline, float)
    if len(polyline) > 1:
        polyline = polyline[np.r_[True, np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-12]]
    if len(polyline) == 1:
        return polyline.copy(), np.array([0.0])
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    # Retaining every corner prevents interpolation between two samples from
    # cutting across a shortcut that was collision checked as separate segments.
    sample_s = np.unique(np.r_[np.arange(0, length[-1], spacing), length])
    return np.column_stack([np.interp(sample_s, length, polyline[:, i]) for i in range(2)]), sample_s


def interpolate_reference(reference, s):
    s = float(np.clip(s, 0, reference["s"][-1]))
    return np.array([np.interp(s, reference["s"], reference["xy"][:, i]) for i in range(2)])


def project_reference(xy, reference, minimum_s=0.0):
    """Closest route point with optional monotone progress restriction."""
    points, length = np.asarray(reference["xy"]), np.asarray(reference["s"])
    if len(points) == 1:
        return 0.0, float(np.linalg.norm(np.asarray(xy) - points[0])), points[0].copy()
    a, delta = points[:-1], np.diff(points, axis=0)
    ds = np.diff(length)
    valid = length[1:] >= minimum_s - 1e-12
    u = np.sum((np.asarray(xy) - a) * delta, axis=1) / np.sum(delta * delta, axis=1)
    lower = np.maximum(0.0, (minimum_s - length[:-1]) / ds)
    u = np.minimum(1.0, np.maximum(lower, u))
    projected = a + u[:, None] * delta
    error = np.linalg.norm(projected - xy, axis=1)
    error[~valid] = np.inf
    index = int(np.argmin(error))
    return float(length[index] + u[index] * ds[index]), float(error[index]), projected[index].copy()


def generate_reference(config):
    """A* on an inflated known map, checked shortcuts, arc-length resampling.

    The time array is nominal cruise timing, not a dynamically feasible cornering
    profile. Unexpected obstacles are deliberately excluded from this function.
    """
    robot, planner = config["robot"], config["planner"]
    start, goal = np.asarray(robot["start"], float), np.asarray(robot["goal"], float)
    known = api.build_scene({"objects": config["world"]["known_objects"]}).objects if config["world"]["known_objects"] else []
    resolution = float(planner["resolution_m"])
    inflation = float(robot["radius_m"] + planner["safety_margin_m"] + planner.get("corner_buffer_m", 0.0))
    height = robot["height_m"]
    xmin, xmax, ymin, ymax = planner["extent"]
    if resolution <= 0 or planner["resample_spacing_m"] <= 0 or planner["reference_speed_mps"] <= 0:
        raise ValueError("Reference resolution, spacing, and nominal speed must be positive")
    xs, ys = np.arange(xmin, xmax + resolution / 2, resolution), np.arange(ymin, ymax + resolution / 2, resolution)
    if not (xmin <= start[0] <= xmax and ymin <= start[1] <= ymax and
            xmin <= goal[0] <= xmax and ymin <= goal[1] <= ymax):
        raise ValueError("Start and goal must lie within planning extent")
    def gap(a, b):
        return swept_surface_gap(a, b, known, inflation, height)
    if gap(start, start) <= 0 or gap(goal, goal) <= 0:
        raise ValueError("Start or goal intersects the inflated known map")
    if np.linalg.norm(start - goal) <= 1e-12:
        route = start[None, :]
    else:
        source = (int(np.argmin(abs(xs - start[0]))), int(np.argmin(abs(ys - start[1]))))
        target = (int(np.argmin(abs(xs - goal[0]))), int(np.argmin(abs(ys - goal[1]))))
        def point(node):
            return np.array([xs[node[0]], ys[node[1]]])
        cache = {}
        def free(node):
            if not (0 <= node[0] < len(xs) and 0 <= node[1] < len(ys)):
                return False
            if node not in cache:
                cache[node] = gap(point(node), point(node)) > 1e-10
            return cache[node]
        if not free(source) or not free(target) or gap(start, point(source)) <= 0 or gap(point(target), goal) <= 0:
            raise ValueError("Nearest start/goal grid nodes are blocked; reduce resolution")
        serial = itertools.count()
        frontier = [(float(np.linalg.norm(point(source) - point(target))), next(serial), source)]
        score, parent, closed = {source: 0.0}, {}, set()
        moves = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while frontier:
            _, _, node = heapq.heappop(frontier)
            if node in closed:
                continue
            if node == target:
                break
            closed.add(node)
            for dx, dy in moves:
                neighbor = (node[0] + dx, node[1] + dy)
                if not free(neighbor) or (dx and dy and (not free((node[0] + dx, node[1])) or not free((node[0], node[1] + dy)))):
                    continue
                if gap(point(node), point(neighbor)) <= 1e-10:
                    continue
                candidate = score[node] + resolution * math.hypot(dx, dy)
                if candidate < score.get(neighbor, math.inf) - 1e-12:
                    score[neighbor], parent[neighbor] = candidate, node
                    heuristic = float(np.linalg.norm(point(neighbor) - point(target)))
                    heapq.heappush(frontier, (candidate + heuristic, next(serial), neighbor))
        if target not in score:
            raise ValueError("No collision-free route exists on the known inflated map")
        nodes, node = [target], target
        while node != source:
            node = parent[node]
            nodes.append(node)
        raw = np.vstack([start, np.array([point(n) for n in nodes[::-1]]), goal])
        # Keep only segments checked against the complete inflated known map.
        route, index = [raw[0]], 0
        while index < len(raw) - 1:
            next_index = len(raw) - 1
            while next_index > index + 1 and gap(raw[index], raw[next_index]) <= 1e-10:
                next_index -= 1
            if gap(raw[index], raw[next_index]) <= 1e-10:
                raise ValueError("A reconstructed reference segment intersects known inflation")
            route.append(raw[next_index])
            index = next_index
        route = np.asarray(route)
    xy, s = resample_route(route, planner["resample_spacing_m"])
    heading = (np.r_[np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])), 0.0] if len(xy) > 1 else
               np.array([np.radians(robot["start_heading_deg"])]))
    if len(xy) > 1:
        heading[-1] = heading[-2]
    speed = np.full(len(xy), planner["reference_speed_mps"], float)
    speed[-1] = 0.0
    return dict(xy=xy, s=s, t=s / planner["reference_speed_mps"], speed=speed, heading=heading,
                raw_xy=route, inflation_m=inflation, known_obstacle_labels=[obj.label for obj in known],
                timing="Nominal t=s/cruise_speed; geometric corners are not dynamically time-scaled")
