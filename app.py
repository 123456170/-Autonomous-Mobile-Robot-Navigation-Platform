import os
import math
import time
import heapq
import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import streamlit.components.v1 as components


# ----------------------------------------------------------------------
# Constants and small utilities
# ----------------------------------------------------------------------

CANVAS_W = 1280
CANVAS_H = 720
DEFAULT_FPS = 10
SIM_DT = 0.15

MODE_LABELS = {
    "amcl": "AMCL (known map)",
    "slam": "Occupancy SLAM (unknown map)",
    "dead": "Dead reckoning",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def r3(value: float) -> float:
    return round(float(value), 3)


# ----------------------------------------------------------------------
# Basic robot / mission data types
# ----------------------------------------------------------------------

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def copy(self) -> "Pose":
        return Pose(self.x, self.y, self.theta)


@dataclass
class Twist:
    vx: float = 0.0
    w: float = 0.0


@dataclass
class Goal:
    name: str
    pose: Pose


@dataclass
class DynamicObstacle:
    name: str
    x: float
    y: float
    radius: float
    waypoints: List[Tuple[float, float]]
    base_speed: float
    speed: float = 0.0
    idx: int = 0
    vx: float = 0.0
    vy: float = 0.0
    active: bool = True

    def update(self, dt: float, speed_scale: float = 1.0) -> None:
        self.speed = self.base_speed * speed_scale
        if not self.active or len(self.waypoints) < 2 or self.speed <= 1e-6:
            self.vx = 0.0
            self.vy = 0.0
            return

        tx, ty = self.waypoints[(self.idx + 1) % len(self.waypoints)]
        dx = tx - self.x
        dy = ty - self.y
        d = math.hypot(dx, dy)

        if d < 0.08:
            self.idx = (self.idx + 1) % len(self.waypoints)
            self.vx = 0.0
            self.vy = 0.0
            return

        nx = dx / d
        ny = dy / d
        self.vx = nx * self.speed
        self.vy = ny * self.speed
        self.x += self.vx * dt
        self.y += self.vy * dt


# ----------------------------------------------------------------------
# Simulation world
# ----------------------------------------------------------------------

class World:
    def __init__(self, scenario: str, dynamic_speed: float, seed: int):
        self.res = 0.1
        self.scenario = scenario
        self.dynamic_speed_scale = dynamic_speed
        self.rng = np.random.default_rng(seed)

        self.cols = 70
        self.rows = 50
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        self.t = 0.0
        self.start = Pose(1.0, 0.55, 0.0)
        self.goals: Dict[str, Pose] = {}
        self.goal_order: List[str] = []
        self.dynamic: List[DynamicObstacle] = []
        self.blockers: List[Dict[str, Any]] = []
        self._static_prob: Optional[np.ndarray] = None

        if scenario == "corridor":
            self._build_corridor()
        elif scenario == "dynamic":
            self._build_dynamic_warehouse()
        else:
            self._build_warehouse()

        self._ensure_free_start()

    def _perimeter(self) -> None:
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _add_rect_cells(self, c0: int, r0: int, c1: int, r1: int, val: int = 1) -> None:
        c0 = clamp(int(c0), 0, self.cols - 1)
        c1 = clamp(int(c1), 0, self.cols - 1)
        r0 = clamp(int(r0), 0, self.rows - 1)
        r1 = clamp(int(r1), 0, self.rows - 1)
        self.grid[r0:r1 + 1, c0:c1 + 1] = val

    def _build_warehouse(self) -> None:
        self.cols = 70
        self.rows = 50
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self._perimeter()

        for c in [15, 35, 55]:
            self._add_rect_cells(c - 1, 8, c + 1, 22)
            self._add_rect_cells(c - 1, 28, c + 1, 42)

        self.start = Pose(1.0, 0.55, 0.0)

        self.goals = {
            "Pick A": Pose(6.3, 1.15, 0.0),
            "Pack B": Pose(6.3, 4.45, math.pi),
            "Charge": Pose(1.0, 4.45, math.pi),
        }
        self.goal_order = ["Pick A", "Pack B", "Charge"]

        self.dynamic = [
            DynamicObstacle("Forklift Alpha", 1.5, 2.5, 0.30, [(1.2, 2.5), (5.0, 2.5)], 0.42),
            DynamicObstacle("Forklift Beta", 6.0, 0.72, 0.26, [(6.2, 0.72), (1.2, 0.72)], 0.34),
        ]

        self.blockers = [
            {"name": "Temporary pallet", "x": 6.2, "y": 2.5, "radius": 0.42,
             "start": 5.0, "end": 12.5, "active": False}
        ]

    def _build_corridor(self) -> None:
        self.cols = 90
        self.rows = 32
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self._perimeter()

        self._add_rect_cells(1, 1, 88, 11)
        self._add_rect_cells(1, 21, 88, 30)

        self._add_rect_cells(30, 12, 31, 15)
        self._add_rect_cells(30, 17, 31, 20)
        self._add_rect_cells(60, 12, 61, 15)
        self._add_rect_cells(60, 17, 61, 20)

        self.start = Pose(1.0, 1.60, 0.0)

        self.goals = {
            "Bay 1": Pose(4.5, 1.60, 0.0),
            "Bay 2": Pose(8.5, 1.60, 0.0),
            "Return": Pose(1.0, 1.60, math.pi),
        }
        self.goal_order = ["Bay 1", "Bay 2", "Return"]

        self.dynamic = [
            DynamicObstacle("Shuttle", 2.0, 1.60, 0.26, [(2.0, 1.60), (8.5, 1.60)], 0.45)
        ]

        self.blockers = [
            {"name": "Door pallet", "x": 5.0, "y": 1.60, "radius": 0.40,
             "start": 4.0, "end": 11.0, "active": False}
        ]

    def _build_dynamic_warehouse(self) -> None:
        self._build_warehouse()

        self.dynamic.extend([
            DynamicObstacle("Cart Gamma", 2.5, 4.45, 0.24, [(2.5, 4.45), (6.0, 4.45)], 0.38),
            DynamicObstacle("Maintenance robot", 4.5, 0.72, 0.22,
                            [(4.5, 0.72), (1.5, 0.72), (1.5, 2.5), (4.5, 2.5)], 0.32),
        ])

        self.blockers.extend([
            {"name": "Falling box", "x": 5.0, "y": 1.15, "radius": 0.38,
             "start": 3.0, "end": 9.0, "active": False},
            {"name": "Cone zone", "x": 2.0, "y": 4.45, "radius": 0.34,
             "start": 10.0, "end": 18.0, "active": False},
        ])

    def _ensure_free_start(self) -> None:
        if not self.collision(self.start, 0.12):
            return

        for dx in np.arange(0.0, 1.2, 0.1):
            for dy in np.arange(-1.0, 1.1, 0.1):
                p = Pose(self.start.x + dx, self.start.y + dy, self.start.theta)
                if 0 <= p.x < self.cols * self.res and 0 <= p.y < self.rows * self.res:
                    if not self.collision(p, 0.12):
                        self.start = p
                        return

    @property
    def static_map_prob(self) -> np.ndarray:
        if self._static_prob is None:
            self._static_prob = np.where(self.grid == 1, 0.95, 0.05).astype(np.float32)
        return self._static_prob

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        c = int(math.floor(x / self.res))
        r = int(math.floor(y / self.res))
        r = clamp(r, 0, self.rows - 1)
        c = clamp(c, 0, self.cols - 1)
        return r, c

    def cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        return (c + 0.5) * self.res, (r + 0.5) * self.res

    def update(self, dt: float) -> None:
        self.t += dt
        for obs in self.dynamic:
            obs.update(dt, self.dynamic_speed_scale)
        for b in self.blockers:
            b["active"] = bool(b["start"] <= self.t <= b["end"])

    def active_obstacles(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in self.dynamic:
            if d.active:
                out.append({
                    "x": d.x, "y": d.y, "radius": d.radius,
                    "vx": d.vx, "vy": d.vy, "name": d.name, "kind": "dynamic",
                })
        for b in self.blockers:
            if b.get("active", False):
                out.append({
                    "x": b["x"], "y": b["y"], "radius": b["radius"],
                    "vx": 0.0, "vy": 0.0, "name": b["name"], "kind": "blocker",
                })
        return out

    def perceived_obstacles(self, noise: float = 0.035) -> List[Dict[str, Any]]:
        out = []
        for obs in self.active_obstacles():
            out.append({
                **obs,
                "x": obs["x"] + float(self.rng.normal(0.0, noise)),
                "y": obs["y"] + float(self.rng.normal(0.0, noise)),
            })
        return out

    def goal_list(self) -> List[Goal]:
        return [Goal(name, self.goals[name]) for name in self.goal_order]

    def _circle_hits_grid(self, x: float, y: float, radius: float) -> bool:
        min_c = max(0, int((x - radius) / self.res))
        max_c = min(self.cols - 1, int((x + radius) / self.res))
        min_r = max(0, int((y - radius) / self.res))
        max_r = min(self.rows - 1, int((y + radius) / self.res))

        r2 = radius * radius
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if self.grid[r, c] != 1:
                    continue
                cx = clamp(x, c * self.res, (c + 1) * self.res)
                cy = clamp(y, r * self.res, (r + 1) * self.res)
                if (x - cx) ** 2 + (y - cy) ** 2 < r2:
                    return True
        return False

    def collision(self, pose: Pose, radius: float) -> bool:
        if self._circle_hits_grid(pose.x, pose.y, radius):
            return True
        for obs in self.active_obstacles():
            if dist2d(pose.x, pose.y, obs["x"], obs["y"]) < radius + obs["radius"]:
                return True
        return False

    def _raycast_with_active(self, pose: Pose, angle: float, max_range: float,
                             active: List[Dict[str, Any]]) -> float:
        step = self.res * 0.45
        x = pose.x
        y = pose.y
        dx = math.cos(angle) * step
        dy = math.sin(angle) * step
        d = 0.0

        while d < max_range:
            x += dx
            y += dy
            d += step

            r, c = self.world_to_cell(x, y)
            if self.grid[r, c] == 1:
                return d

            for obs in active:
                if (x - obs["x"]) ** 2 + (y - obs["y"]) ** 2 < obs["radius"] ** 2:
                    return d

        return max_range

    def scan(self, pose: Pose, n_rays: int = 60, max_range: float = 5.0,
             noise: float = 0.03, drop_prob: float = 0.02):
        active = self.active_obstacles()
        angles = pose.theta + np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
        ranges = np.empty(n_rays, dtype=np.float32)
        valid = np.ones(n_rays, dtype=bool)

        for i, a in enumerate(angles):
            r = self._raycast_with_active(pose, float(a), max_range, active)
            if float(self.rng.random()) < drop_prob:
                valid[i] = False
                ranges[i] = max_range
            else:
                r = float(np.clip(r + self.rng.normal(0.0, noise), 0.05, max_range))
                ranges[i] = r

        return ranges, angles, valid

    def camera_detections(self, pose: Pose, max_range: float = 4.0, n_beams: int = 9):
        active = self.active_obstacles()
        bearings = np.linspace(-0.55, 0.55, n_beams)
        dets: List[Dict[str, Any]] = []

        for b in bearings:
            a = pose.theta + float(b)
            r = self._raycast_with_active(pose, a, max_range, active)
            r_noisy = float(np.clip(r + self.rng.normal(0.0, 0.04), 0.1, max_range))

            conf = clamp(1.0 - r_noisy / max_range, 0.05, 0.98)
            if r_noisy >= max_range * 0.97:
                conf *= 0.35

            kind = "free"
            ex = pose.x + math.cos(a) * r_noisy
            ey = pose.y + math.sin(a) * r_noisy
            for obs in active:
                if dist2d(ex, ey, obs["x"], obs["y"]) < obs["radius"] + 0.12:
                    kind = obs["kind"]
                    break

            dets.append({
                "bearing": float(b),
                "range": r_noisy,
                "confidence": float(conf),
                "type": kind,
            })

        return dets


# ----------------------------------------------------------------------
# Occupancy grid
# ----------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, rows: int, cols: int, res: float,
                 known_grid: Optional[np.ndarray] = None):
        self.rows = rows
        self.cols = cols
        self.res = res
        self.log_odds = np.zeros((rows, cols), dtype=np.float32)
        self.explored = np.zeros((rows, cols), dtype=bool)
        self.prob: Optional[np.ndarray] = None
        self.dirty = True

        if known_grid is not None:
            self.log_odds[known_grid == 1] = 5.5
            self.log_odds[known_grid == 0] = -2.5
            self.explored[:] = True

    def is_empty(self) -> bool:
        return bool(np.max(np.abs(self.log_odds)) < 0.05)

    def probability(self) -> np.ndarray:
        if self.dirty or self.prob is None:
            self.prob = 1.0 / (1.0 + np.exp(-self.log_odds))
            self.prob = self.prob.astype(np.float32)
            self.dirty = False
        return self.prob

    def _mark(self, x: float, y: float, occupied: bool) -> None:
        c = int(math.floor(x / self.res))
        r = int(math.floor(y / self.res))
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return

        if occupied:
            self.log_odds[r, c] = clamp(self.log_odds[r, c] + 0.75, -6.0, 6.0)
        else:
            self.log_odds[r, c] = clamp(self.log_odds[r, c] - 0.22, -6.0, 6.0)

        self.explored[r, c] = True
        self.dirty = True

    def update(self, pose: Pose, ranges: np.ndarray, angles: np.ndarray,
               valid: np.ndarray, max_range: float) -> None:
        self._mark(pose.x, pose.y, False)
        step = self.res * 0.8

        for r, a, v in zip(ranges, angles, valid):
            if not v:
                continue

            r = float(r)
            a = float(a)
            ca = math.cos(a)
            sa = math.sin(a)

            if r < max_range * 0.95:
                d = 0.18
                while d < r - 0.12 and d < max_range:
                    self._mark(pose.x + ca * d, pose.y + sa * d, False)
                    d += step
                self._mark(pose.x + ca * r, pose.y + sa * r, True)
            else:
                d = 0.18
                while d < max_range:
                    self._mark(pose.x + ca * d, pose.y + sa * d, False)
                    d += step


# ----------------------------------------------------------------------
# Particle filter localization
# ----------------------------------------------------------------------

class ParticleFilter:
    def __init__(self, n_particles: int, initial_pose: Pose, spread: float,
                 rng: np.random.Generator):
        self.n = int(n_particles)
        self.rng = rng
        self.spread = spread

        self.particles = np.zeros((self.n, 3), dtype=np.float32)
        self.particles[:, 0] = initial_pose.x + rng.normal(0.0, spread, self.n)
        self.particles[:, 1] = initial_pose.y + rng.normal(0.0, spread, self.n)
        self.particles[:, 2] = initial_pose.theta + rng.normal(0.0, spread * 0.45, self.n)
        self.particles[:, 2] = np.array([wrap_pi(t) for t in self.particles[:, 2]], dtype=np.float32)

        self.weights = np.ones(self.n, dtype=np.float32) / self.n
        self.ess = float(self.n)
        self.sample_indices: Optional[np.ndarray] = None

    def motion_update(self, delta: Twist, noise_xy: float = 0.015, noise_th: float = 0.012) -> None:
        ds = float(delta.vx)
        dth = float(delta.w)

        for i in range(self.n):
            th = float(self.particles[i, 2])
            self.particles[i, 0] += ds * math.cos(th) + float(self.rng.normal(0.0, noise_xy))
            self.particles[i, 1] += ds * math.sin(th) + float(self.rng.normal(0.0, noise_xy))
            self.particles[i, 2] = wrap_pi(th + dth + float(self.rng.normal(0.0, noise_th)))

    def _prob_at(self, x, y, map_prob, res, rows, cols) -> float:
        c = int(math.floor(x / res))
        r = int(math.floor(y / res))
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 0.95
        return float(map_prob[r, c])

    def sensor_update(self, ranges, angles, valid, map_prob, res, max_range) -> None:
        n_rays = len(ranges)
        if n_rays == 0:
            return

        if self.sample_indices is None or len(self.sample_indices) != n_rays:
            sample_count = min(12, n_rays)
            self.sample_indices = np.linspace(0, n_rays - 1, sample_count, dtype=int)

        rows, cols = map_prob.shape
        logw = np.zeros(self.n, dtype=np.float64)

        for i in range(self.n):
            px = float(self.particles[i, 0])
            py = float(self.particles[i, 1])
            ll = 0.0

            for j in self.sample_indices:
                if not valid[j]:
                    continue

                r = float(ranges[j])
                a = float(angles[j])
                ca = math.cos(a)
                sa = math.sin(a)
                ray_ll = 0.0

                if r < max_range * 0.95:
                    for frac in (0.25, 0.55, 0.85):
                        p = self._prob_at(px + ca * r * frac, py + sa * r * frac, map_prob, res, rows, cols)
                        ray_ll += math.log(max(1.0 - p, 0.08))
                    p = self._prob_at(px + ca * r, py + sa * r, map_prob, res, rows, cols)
                    ray_ll += math.log(max(p, 0.08))
                else:
                    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                        p = self._prob_at(px + ca * r * frac, py + sa * r * frac, map_prob, res, rows, cols)
                        ray_ll += math.log(max(1.0 - p, 0.08))

                ll += max(ray_ll, -5.0)

            logw[i] = ll

        max_log = float(np.max(logw))
        if not math.isfinite(max_log):
            self.weights = np.ones(self.n, dtype=np.float32) / self.n
        else:
            w = np.exp(logw - max_log)
            s = float(np.sum(w))
            if s <= 1e-12:
                w = np.ones(self.n, dtype=np.float64)
                s = float(self.n)
            self.weights = (w / s).astype(np.float32)

        self.ess = float(1.0 / np.sum(self.weights ** 2))
        if self.ess < 0.5 * self.n:
            self._resample_systematic()

    def _resample_systematic(self) -> None:
        pos = (float(self.rng.random()) + np.arange(self.n)) / self.n
        cum = np.cumsum(self.weights)
        cum[-1] = 1.0
        idx = np.searchsorted(cum, pos)
        idx = np.clip(idx, 0, self.n - 1)
        self.particles = self.particles[idx, :].copy()
        self.weights = np.ones(self.n, dtype=np.float32) / self.n
        self.ess = float(self.n)

    def estimate(self) -> Tuple[Pose, np.ndarray, float]:
        w = self.weights
        x = float(np.sum(w * self.particles[:, 0]))
        y = float(np.sum(w * self.particles[:, 1]))

        cos_sum = float(np.sum(w * np.cos(self.particles[:, 2])))
        sin_sum = float(np.sum(w * np.sin(self.particles[:, 2])))
        theta = float(self.particles[0, 2]) if math.hypot(cos_sum, sin_sum) < 1e-9 else math.atan2(sin_sum, cos_sum)
        theta = wrap_pi(theta)

        var_x = float(np.sum(w * (self.particles[:, 0] - x) ** 2))
        var_y = float(np.sum(w * (self.particles[:, 1] - y) ** 2))
        dth = np.array([wrap_pi(float(t) - theta) for t in self.particles[:, 2]], dtype=np.float32)
        var_th = float(np.sum(w * dth ** 2))

        cov = np.diag([max(var_x, 1e-6), max(var_y, 1e-6), max(var_th, 1e-6)]).astype(np.float32)
        return Pose(x, y, theta), cov, float(self.ess)


# ----------------------------------------------------------------------
# Robot model
# ----------------------------------------------------------------------

class Robot:
    def __init__(self, pose: Pose, params: Dict[str, Any], rng: np.random.Generator):
        self.true_pose = pose.copy()
        self.odom_pose = pose.copy()
        self.prev_true = pose.copy()
        self.prev_odom = pose.copy()

        self.twist = Twist()
        self.odom_delta = Twist()

        self.radius = float(params["robot_radius"])
        self.max_v = float(params["max_vel"])
        self.max_w = float(params["max_vel"]) * 1.8
        self.max_acc = float(params["max_acc"])
        self.odom_noise = float(params["odom_noise"])
        self.rng = rng

    def update(self, cmd: Twist, dt: float) -> None:
        self.prev_true = self.true_pose.copy()
        self.prev_odom = self.odom_pose.copy()

        desired_v = clamp(float(cmd.vx), -self.max_v, self.max_v)
        desired_w = clamp(float(cmd.w), -self.max_w, self.max_w)

        dv = clamp(desired_v - self.twist.vx, -self.max_acc * dt, self.max_acc * dt)
        dw = clamp(desired_w - self.twist.w, -self.max_acc * 1.6 * dt, self.max_acc * 1.6 * dt)

        self.twist.vx += dv
        self.twist.w += dw

        th = self.true_pose.theta
        self.true_pose.x += self.twist.vx * math.cos(th) * dt
        self.true_pose.y += self.twist.vx * math.sin(th) * dt
        self.true_pose.theta = wrap_pi(th + self.twist.w * dt)

        ov = self.twist.vx + float(self.rng.normal(0.0, self.odom_noise))
        ow = self.twist.w + float(self.rng.normal(0.0, self.odom_noise * 0.7))

        oth = self.odom_pose.theta
        self.odom_pose.x += ov * math.cos(oth) * dt
        self.odom_pose.y += ov * math.sin(oth) * dt
        self.odom_pose.theta = wrap_pi(oth + ow * dt)

        self.odom_delta = Twist(ov * dt, ow * dt)


# ----------------------------------------------------------------------
# Costmap, global planner, local planner
# ----------------------------------------------------------------------

def build_costmap(world, occ_prob, obstacles, safety_margin, robot_radius, mode,
                  explored=None) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "slam":
        if explored is None:
            explored = np.ones_like(occ_prob, dtype=bool)
        base = (occ_prob > 0.60) | (~explored)
    else:
        base = world.grid == 1

    arr = base.astype(np.uint8).copy()

    for obs in obstacles:
        c = int(math.floor(obs["x"] / world.res))
        r = int(math.floor(obs["y"] / world.res))
        rad = int((obs["radius"] + robot_radius + safety_margin * 0.5) / world.res) + 1
        cv2.circle(arr, (c, r), max(1, rad), 1, -1)

    inflate = max(1, int((robot_radius + safety_margin * 0.5) / world.res))
    kernel = np.ones((2 * inflate + 1, 2 * inflate + 1), dtype=np.uint8)
    blocked = cv2.dilate(arr, kernel, iterations=1) > 0

    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    dist = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3) * world.res
    return blocked, dist.astype(np.float32)


def nearest_free_cell(blocked, rc, max_rad=12):
    rows, cols = blocked.shape
    r0 = clamp(rc[0], 0, rows - 1)
    c0 = clamp(rc[1], 0, cols - 1)

    if not blocked[r0, c0]:
        return r0, c0

    for rad in range(1, max_rad + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r = r0 + dr
                c = c0 + dc
                if 0 <= r < rows and 0 <= c < cols and not blocked[r, c]:
                    return r, c
    return None


def astar(blocked, start, goal):
    rows, cols = blocked.shape
    if start == goal:
        return [start]

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(rc):
        return math.hypot(rc[0] - goal[0], rc[1] - goal[1])

    g = np.full((rows, cols), np.inf, dtype=np.float32)
    g[start] = 0.0
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    open_heap = [(heuristic(start), 0.0, start)]

    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return list(reversed(path))

        if gc > g[cur] + 1e-6:
            continue

        r, c = cur
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            if blocked[nr, nc]:
                continue

            ng = gc + math.hypot(dr, dc)
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + heuristic((nr, nc)), ng, (nr, nc)))

    return None


def plan_global(blocked, world, start: Pose, goal: Pose):
    start_rc = nearest_free_cell(blocked, world.world_to_cell(start.x, start.y))
    goal_rc = nearest_free_cell(blocked, world.world_to_cell(goal.x, goal.y))

    if start_rc is None or goal_rc is None:
        return None

    cells = astar(blocked, start_rc, goal_rc)
    if cells is None:
        return None

    return [Pose(*world.cell_to_world(r, c), 0.0) for r, c in cells]


class LocalPlanner:
    def __init__(self, params: Dict[str, Any]):
        self.max_v = float(params["max_vel"])
        self.max_w = self.max_v * 1.8
        self.v_bins = 6
        self.w_bins = 7
        self.horizon_time = 1.0

    def _select_target(self, pose, path, goal):
        if not path:
            return goal.pose, True

        for p in path:
            if dist2d(pose.x, pose.y, p.x, p.y) > 0.40:
                return p, False
        return path[-1], True

    def _static_clearance(self, pose, blocked, dist_transform, res):
        c = int(math.floor(pose.x / res))
        r = int(math.floor(pose.y / res))
        rows, cols = blocked.shape
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 0.0
        if blocked[r, c]:
            return 0.0
        return float(dist_transform[r, c])

    def _dynamic_clearance(self, pose, obstacles, tau, robot_radius):
        if not obstacles:
            return 10.0
        min_clear = 10.0
        for obs in obstacles:
            ox = obs["x"] + obs.get("vx", 0.0) * tau
            oy = obs["y"] + obs.get("vy", 0.0) * tau
            min_clear = min(min_clear, dist2d(pose.x, pose.y, ox, oy) - obs["radius"] - robot_radius)
        return min_clear

    def compute(self, pose, goal, path, blocked, dist_transform, obstacles,
                robot_twist, world_res, robot_radius, safety_margin, dt):
        target, final = self._select_target(pose, path, goal)

        if final and dist2d(pose.x, pose.y, goal.pose.x, goal.pose.y) < 0.55:
            target_theta = goal.pose.theta
        else:
            target_theta = math.atan2(target.y - pose.y, target.x - pose.x)

        v_samples = np.linspace(0.0, self.max_v, self.v_bins)
        w_samples = np.linspace(-self.max_w, self.max_w, self.w_bins)
        horizon_steps = max(3, int(self.horizon_time / dt))

        best_cost = float("inf")
        best_cmd = Twist(0.0, 0.0)
        found = False

        for v in v_samples:
            for w in w_samples:
                p = pose.copy()
                safe = True
                min_static = 10.0
                min_dyn = 10.0

                for k in range(horizon_steps):
                    tau = (k + 1) * dt
                    p.x += v * math.cos(p.theta) * dt
                    p.y += v * math.sin(p.theta) * dt
                    p.theta = wrap_pi(p.theta + w * dt)

                    sd = self._static_clearance(p, blocked, dist_transform, world_res)
                    dd = self._dynamic_clearance(p, obstacles, tau, robot_radius)

                    if sd <= 0.0 or dd <= safety_margin * 0.25:
                        safe = False
                        break

                    min_static = min(min_static, sd)
                    min_dyn = min(min_dyn, dd)

                if not safe:
                    continue

                heading_err = abs(wrap_pi(target_theta - p.theta))
                dist_err = dist2d(target.x, target.y, p.x, p.y)

                cost = (1.0 * dist_err + 1.2 * heading_err
                        + 0.15 * abs(v - self.max_v * 0.75) + 0.35 * abs(w)
                        - 1.5 * min(min_static, 1.0) - 1.0 * min(min_dyn, 1.0))

                if cost < best_cost:
                    best_cost = cost
                    best_cmd = Twist(float(v), float(w))
                    found = True

        if found:
            return best_cmd, True

        best_rot_cost = float("inf")
        best_rot_w = 0.0
        found_rot = False

        sd0 = self._static_clearance(pose, blocked, dist_transform, world_res)
        dd0 = self._dynamic_clearance(pose, obstacles, 0.0, robot_radius)

        if sd0 > 0.0 and dd0 > safety_margin * 0.25:
            for w in w_samples:
                cost = abs(wrap_pi(target_theta - pose.theta)) + 0.2 * abs(w)
                if cost < best_rot_cost:
                    best_rot_cost = cost
                    best_rot_w = float(w)
                    found_rot = True

        if found_rot:
            return Twist(0.0, best_rot_w), True

        return Twist(0.0, 0.0), False


# ----------------------------------------------------------------------
# Navigation state machine
# ----------------------------------------------------------------------

class Navigator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.local = LocalPlanner(params)

        self.goal_queue: deque = deque()
        self.current_goal: Optional[Goal] = None
        self.path: List[Pose] = []

        self.state = "IDLE"
        self.replan_timer = 0.0
        self.stuck = 0
        self.recoveries = 0
        self.safety_stops = 0
        self.completed: List[Dict[str, Any]] = []
        self.time = 0.0

    def set_goals(self, goals: List[Goal]) -> None:
        self.goal_queue = deque(goals)
        self.current_goal = None
        self.path = []
        self.state = "PLANNING"
        self.replan_timer = 0.0
        self.stuck = 0
        self.recoveries = 0
        self.safety_stops = 0
        self.completed = []
        self.time = 0.0

    def add_goal(self, goal: Goal) -> None:
        self.goal_queue.append(goal)
        if self.state in ("IDLE", "ALL_GOALS_COMPLETE"):
            self.state = "PLANNING"

    def _path_blocked(self, pose, blocked, world) -> bool:
        if not self.path:
            return True
        for p in self.path[:6]:
            r, c = world.world_to_cell(p.x, p.y)
            if blocked[r, c]:
                return True
        return False

    def compute(self, est_pose, blocked, dist_transform, world, obstacles,
                robot_twist, sensor_stats, dt) -> Twist:
        self.time += dt

        if self.current_goal is None:
            if len(self.goal_queue) == 0:
                self.state = "ALL_GOALS_COMPLETE" if self.completed else "IDLE"
                return Twist(0.0, 0.0)

            self.current_goal = self.goal_queue.popleft()
            self.path = []
            self.replan_timer = 0.0
            self.state = "PLANNING"

        unreliable = (
            sensor_stats["ess_frac"] < self.params["min_ess"]
            or sensor_stats["cov_trace"] > self.params["max_cov"]
            or sensor_stats["lidar_valid"] < self.params["min_lidar_valid"]
        )

        if unreliable:
            if self.state != "SAFETY_STOP":
                self.safety_stops += 1
            self.state = "SAFETY_STOP"
            return Twist(0.0, 0.0)

        goal_pose = self.current_goal.pose
        dgoal = dist2d(est_pose.x, est_pose.y, goal_pose.x, goal_pose.y)

        if dgoal < self.params["xy_tolerance"]:
            self.completed.append({"name": self.current_goal.name, "time": self.time})
            self.current_goal = None
            self.path = []
            self.state = "GOAL_REACHED"
            return Twist(0.0, 0.0)

        self.replan_timer -= dt
        if self.replan_timer <= 0.0 or len(self.path) == 0 or self._path_blocked(est_pose, blocked, world):
            path = plan_global(blocked, world, est_pose, goal_pose)
            if path is None:
                self.state = "RECOVERY"
                self.recoveries += 1
                self.replan_timer = 1.0
                target_theta = math.atan2(goal_pose.y - est_pose.y, goal_pose.x - est_pose.x)
                w = clamp(wrap_pi(target_theta - est_pose.theta),
                          -self.params["max_vel"] * 1.8, self.params["max_vel"] * 1.8)
                return Twist(0.0, w)

            self.path = path
            self.replan_timer = 2.0
            self.state = "FOLLOWING"

        cmd, safe = self.local.compute(
            est_pose, self.current_goal, self.path, blocked, dist_transform,
            obstacles, robot_twist, world.res, self.params["robot_radius"],
            self.params["safety_margin"], dt,
        )

        if not safe:
            self.stuck += 1
            if self.stuck > 5:
                self.recoveries += 1
                self.stuck = 0
                self.replan_timer = 0.0
                self.state = "RECOVERY"
                target = self.path[0] if self.path else goal_pose
                target_theta = math.atan2(target.y - est_pose.y, target.x - est_pose.x)
                w = clamp(wrap_pi(target_theta - est_pose.theta),
                          -self.params["max_vel"] * 1.8, self.params["max_vel"] * 1.8)
                return Twist(0.0, w)
            return Twist(0.0, 0.0)

        self.stuck = 0
        if self.state == "RECOVERY":
            self.state = "FOLLOWING"

        return cmd


# ----------------------------------------------------------------------
# Integrated simulation
# ----------------------------------------------------------------------

class NavigationSim:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.mode = params["mode"]
        self.rng = np.random.default_rng(params["seed"])

        self.world = World(params["scenario"], params["dynamic_speed"], params["seed"])
        self.robot = Robot(self.world.start, params, self.rng)

        known_grid = self.world.grid if self.mode != "slam" else None
        self.occ = OccupancyGrid(self.world.rows, self.world.cols, self.world.res, known_grid=known_grid)

        spread = 0.12 if self.mode == "amcl" else 0.25
        self.pf = ParticleFilter(params["particles"], self.world.start, spread, self.rng) if self.mode != "dead" else None
        self.nav = Navigator(params)

        self.last_cmd = Twist()
        self.step = 0
        self.t = 0.0
        self.collisions = 0
        self.travel_distance = 0.0

        self.estimate = self.world.start.copy()
        self.cov = np.eye(3, dtype=np.float32) * 0.01
        self.ess = float(params["particles"])

        self.latest_scan: Optional[np.ndarray] = None
        self.latest_angles: Optional[np.ndarray] = None
        self.latest_valid: Optional[np.ndarray] = None
        self.latest_camera: List[Dict[str, Any]] = []
        self.perceived_obs: List[Dict[str, Any]] = []

        self.occ_prob = self.world.static_map_prob if self.mode != "slam" else self.occ.probability()
        self.latest_blocked, self.latest_dist = build_costmap(
            self.world, self.occ_prob, self.perceived_obs,
            self.params["safety_margin"], self.robot.radius, self.mode,
            self.occ.explored if self.mode == "slam" else None,
        )

        self.sensor_stats = {
            "lidar_valid": 1.0, "camera_conf": 0.0, "ess_frac": 1.0,
            "cov_trace": 0.01, "obstacle_points": 0,
        }

        self.telemetry: List[Dict[str, Any]] = []
        self.latest_frame: Optional[np.ndarray] = None

    def step_once(self, render: bool = True):
        dt = SIM_DT
        self.world.update(dt)
        self.robot.update(self.last_cmd, dt)

        collided = self.world.collision(self.robot.true_pose, self.robot.radius)
        if collided:
            self.collisions += 1
            self.robot.true_pose = self.robot.prev_true.copy()
            self.robot.odom_pose = self.robot.prev_odom.copy()
            self.robot.twist = Twist(0.0, 0.0)
            self.robot.odom_delta = Twist(0.0, 0.0)
            self.last_cmd = Twist(0.0, 0.0)
            self.nav.state = "RECOVERY"
            self.nav.stuck = 6
            self.nav.recoveries += 1
        else:
            self.travel_distance += dist2d(
                self.robot.true_pose.x, self.robot.true_pose.y,
                self.robot.prev_true.x, self.robot.prev_true.y,
            )

        ranges, angles, valid = self.world.scan(
            self.robot.true_pose, n_rays=60, max_range=5.0,
            noise=self.params["lidar_noise"], drop_prob=self.params["lidar_drop"],
        )
        camera = self.world.camera_detections(self.robot.true_pose, max_range=4.0)

        self.latest_scan = ranges
        self.latest_angles = angles
        self.latest_valid = valid
        self.latest_camera = camera
        self.perceived_obs = self.world.perceived_obstacles(noise=0.035)

        if self.mode == "dead":
            self.estimate = self.robot.odom_pose.copy()
            growth = 0.0009 * self.t
            self.cov = np.diag([0.01 + growth, 0.01 + growth, 0.004 + growth * 0.5]).astype(np.float32)
            self.ess = 1.0
            self.occ_prob = self.world.static_map_prob
            ess_frac = 1.0
            cov_trace = float(np.trace(self.cov))
        else:
            assert self.pf is not None
            self.pf.motion_update(self.robot.odom_delta,
                                  noise_xy=self.params["process_noise"],
                                  noise_th=self.params["process_noise"] * 0.8)

            if self.mode == "slam" and self.occ.is_empty():
                self.occ.update(self.robot.odom_pose, ranges, angles, valid, 5.0)

            map_prob = self.occ.probability() if self.mode == "slam" else self.world.static_map_prob
            self.pf.sensor_update(ranges, angles, valid, map_prob, self.world.res, 5.0)

            est, cov, ess = self.pf.estimate()
            self.estimate = est
            self.cov = cov
            self.ess = ess

            if self.mode == "slam":
                self.occ.update(est, ranges, angles, valid, 5.0)
                self.occ_prob = self.occ.probability()
            else:
                self.occ_prob = self.world.static_map_prob

            ess_frac = self.ess / max(1, self.params["particles"])
            cov_trace = float(np.trace(self.cov))

        valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
        cam_conf = float(np.mean([d["confidence"] for d in camera])) if camera else 0.0
        obstacle_points = int(np.sum((ranges < 4.75) & valid))

        self.sensor_stats = {
            "lidar_valid": valid_ratio,
            "camera_conf": cam_conf,
            "ess_frac": ess_frac,
            "cov_trace": cov_trace,
            "obstacle_points": obstacle_points,
        }

        self.latest_blocked, self.latest_dist = build_costmap(
            self.world, self.occ_prob, self.perceived_obs,
            self.params["safety_margin"], self.robot.radius, self.mode,
            self.occ.explored if self.mode == "slam" else None,
        )

        cmd = self.nav.compute(
            self.estimate, self.latest_blocked, self.latest_dist, self.world,
            self.perceived_obs, self.robot.twist, self.sensor_stats, dt,
        )
        self.last_cmd = cmd

        err = dist2d(self.estimate.x, self.estimate.y, self.robot.true_pose.x, self.robot.true_pose.y)

        self.telemetry.append({
            "t": round(self.t, 2),
            "state": self.nav.state,
            "true_x": round(self.robot.true_pose.x, 3),
            "true_y": round(self.robot.true_pose.y, 3),
            "est_x": round(self.estimate.x, 3),
            "est_y": round(self.estimate.y, 3),
            "error_m": round(err, 3),
            "ess": round(ess_frac, 2),
            "cov": round(cov_trace, 3),
            "lidar_valid": round(valid_ratio, 2),
            "camera_conf": round(cam_conf, 2),
            "vx": round(cmd.vx, 2),
            "w": round(cmd.w, 2),
            "goal": self.nav.current_goal.name if self.nav.current_goal else "-",
        })

        self.step += 1
        self.t += dt

        if render:
            self.latest_frame = self.render()

        return self.latest_frame

    def export_frame(self) -> Dict[str, Any]:
        rays: List[List[float]] = []
        if self.latest_scan is not None and self.latest_angles is not None and self.latest_valid is not None:
            tp = self.robot.true_pose
            for i in range(0, len(self.latest_scan), 2):
                if not self.latest_valid[i]:
                    continue
                rr = float(self.latest_scan[i])
                aa = float(self.latest_angles[i])
                ex = tp.x + rr * math.cos(aa)
                ey = tp.y + rr * math.sin(aa)
                rays.append([r3(ex), r3(ey), 1 if rr < 4.75 else 0])

        err = dist2d(self.estimate.x, self.estimate.y, self.robot.true_pose.x, self.robot.true_pose.y)

        dyn = [{"x": r3(o["x"]), "y": r3(o["y"]), "r": r3(o["radius"]), "name": o["name"]}
               for o in self.world.active_obstacles() if o["kind"] == "dynamic"]
        block = [{"x": r3(o["x"]), "y": r3(o["y"]), "r": r3(o["radius"]), "name": o["name"]}
                 for o in self.world.active_obstacles() if o["kind"] == "blocker"]

        return {
            "t": r3(self.t),
            "state": self.nav.state,
            "true": [r3(self.robot.true_pose.x), r3(self.robot.true_pose.y), r3(self.robot.true_pose.theta)],
            "est": [r3(self.estimate.x), r3(self.estimate.y), r3(self.estimate.theta)],
            "path": [[r3(p.x), r3(p.y)] for p in self.nav.path],
            "rays": rays,
            "dyn": dyn,
            "block": block,
            "metrics": {
                "err": r3(err),
                "ess": round(float(self.sensor_stats["ess_frac"]), 2),
                "cov": round(float(self.sensor_stats["cov_trace"]), 3),
                "lidar": round(float(self.sensor_stats["lidar_valid"]), 2),
                "cam": round(float(self.sensor_stats["camera_conf"]), 2),
                "vx": round(float(self.last_cmd.vx), 2),
                "w": round(float(self.last_cmd.w), 2),
                "collisions": int(self.collisions),
                "recoveries": int(self.nav.recoveries),
                "safety": int(self.nav.safety_stops),
                "travel": round(float(self.travel_distance), 2),
                "goal": self.nav.current_goal.name if self.nav.current_goal else "-",
                "completed": len(self.nav.completed),
            },
        }

    def summary(self) -> Dict[str, Any]:
        errors = [row["error_m"] for row in self.telemetry]
        return {
            "state": self.nav.state,
            "sim_time_s": round(self.t, 2),
            "goals_completed": len(self.nav.completed),
            "collisions": int(self.collisions),
            "recoveries": int(self.nav.recoveries),
            "safety_stops": int(self.nav.safety_stops),
            "travel_distance_m": round(float(self.travel_distance), 2),
            "mean_loc_error_m": round(float(np.mean(errors)), 3) if errors else 0.0,
            "final_loc_error_m": round(float(errors[-1]), 3) if errors else 0.0,
        }

    # ---------------- 2D top-down render (dashboard + MP4 recorder) ----------------
    def render(self) -> np.ndarray:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        cv2.putText(canvas, "Autonomous Mobile Robot Navigation - Live Simulation (top-down recorder view)",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (12, 48), (735, 610), (65, 65, 65), 1)

        map_x, map_y = 20, 60
        max_w, max_h = 700, 540
        world_w_m = self.world.cols * self.world.res
        world_h_m = self.world.rows * self.world.res
        scale = min(max_w / world_w_m, max_h / world_h_m)

        draw_w = int(world_w_m * scale)
        draw_h = int(world_h_m * scale)

        wall_mask = self.occ_prob > 0.60 if self.mode == "slam" else self.world.grid == 1

        vis = np.zeros((self.world.rows, self.world.cols, 3), dtype=np.uint8)
        vis[:] = (42, 42, 42)
        vis[wall_mask] = (190, 190, 190)
        vis = cv2.resize(vis, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
        canvas[map_y:map_y + draw_h, map_x:map_x + draw_w] = vis

        def w2p(x, y):
            return int(map_x + x * scale), int(map_y + y * scale)

        if len(self.nav.path) > 1:
            pts = np.array([[w2p(p.x, p.y)] for p in self.nav.path], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 190, 255), 2, cv2.LINE_AA)

        current_name = self.nav.current_goal.name if self.nav.current_goal else None
        for g in self.world.goal_list():
            pt = w2p(g.pose.x, g.pose.y)
            color = (0, 255, 255) if g.name == current_name else (180, 180, 0)
            cv2.drawMarker(canvas, pt, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.putText(canvas, g.name, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        for obs in self.world.active_obstacles():
            center = w2p(obs["x"], obs["y"])
            radius = max(2, int(obs["radius"] * scale))
            color = (0, 140, 255) if obs["kind"] == "blocker" else (0, 0, 255)
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, (255, 255, 255), 1, cv2.LINE_AA)

        for obs in self.perceived_obs:
            center = w2p(obs["x"], obs["y"])
            radius = max(2, int(obs["radius"] * scale))
            cv2.circle(canvas, center, radius + 2, (0, 255, 0), 1, cv2.LINE_AA)

        if self.latest_scan is not None and self.latest_angles is not None and self.latest_valid is not None:
            est = self.estimate
            p0 = w2p(est.x, est.y)
            for i in range(0, len(self.latest_scan), 2):
                if not self.latest_valid[i]:
                    continue
                r = float(self.latest_scan[i])
                a = float(self.latest_angles[i])
                p1 = w2p(est.x + r * math.cos(a), est.y + r * math.sin(a))
                color = (0, 220, 0) if r < 4.75 else (90, 90, 90)
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

        def draw_pose_triangle(pose, color, size=9):
            x, y = w2p(pose.x, pose.y)
            pts = []
            for ang in [pose.theta, pose.theta + 2.35, pose.theta - 2.35]:
                pts.append([int(x + size * math.cos(ang)), int(y + size * math.sin(ang))])
            cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], color)
            cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        draw_pose_triangle(self.robot.true_pose, (0, 255, 0), 10)
        draw_pose_triangle(self.estimate, (255, 180, 0), 8)
        draw_pose_triangle(self.robot.odom_pose, (0, 255, 255), 6)

        panel_x = 755
        cv2.rectangle(canvas, (panel_x - 10, 48), (1270, 685), (70, 70, 70), 1)

        state_colors = {
            "IDLE": (180, 180, 180), "PLANNING": (255, 180, 0), "FOLLOWING": (0, 220, 0),
            "RECOVERY": (0, 140, 255), "SAFETY_STOP": (0, 0, 255),
            "GOAL_REACHED": (255, 255, 0), "ALL_GOALS_COMPLETE": (255, 0, 0),
        }
        cv2.putText(canvas, f"STATE: {self.nav.state}", (panel_x, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, state_colors.get(self.nav.state, (220, 220, 220)), 2, cv2.LINE_AA)

        err = dist2d(self.estimate.x, self.estimate.y, self.robot.true_pose.x, self.robot.true_pose.y)

        lines = [
            f"Scenario: {self.params.get('scenario_label', self.world.scenario)}",
            f"Mode: {self.params.get('mode_label', self.mode)}",
            f"Sim time: {self.t:5.1f}s   Step: {self.step}",
            f"Travel distance: {self.travel_distance:5.2f} m",
            "",
            f"True pose : ({self.robot.true_pose.x:5.2f}, {self.robot.true_pose.y:5.2f})",
            f"Est pose  : ({self.estimate.x:5.2f}, {self.estimate.y:5.2f})",
            f"Loc error : {err:5.3f} m",
            f"ESS frac  : {self.sensor_stats['ess_frac']:5.2f}   Cov: {self.sensor_stats['cov_trace']:5.3f}",
            "",
            f"Lidar valid: {self.sensor_stats['lidar_valid'] * 100:5.1f}%",
            f"Camera conf: {self.sensor_stats['camera_conf'] * 100:5.1f}%",
            f"Cmd vx={self.last_cmd.vx:5.2f}  w={self.last_cmd.w:5.2f}",
            "",
            f"Current goal : {self.nav.current_goal.name if self.nav.current_goal else '-'}",
            f"Queued goals : {len(self.nav.goal_queue)}",
            f"Completed    : {len(self.nav.completed)}",
            f"Collisions   : {self.collisions}",
            f"Recoveries   : {self.nav.recoveries}",
            f"Safety stops : {self.nav.safety_stops}",
        ]

        y = 104
        for line in lines:
            cv2.putText(canvas, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
            y += 20

        return canvas


# ----------------------------------------------------------------------
# 3D Gazebo-style scene builder
# ----------------------------------------------------------------------

def build_scene(sim: NavigationSim) -> Dict[str, Any]:
    w = sim.world
    rng = np.random.default_rng(991)
    W = w.cols * w.res
    H = w.rows * w.res

    cub: List[Dict[str, Any]] = []
    marks: List[Dict[str, Any]] = []

    wall_h = 2.0
    t = 0.12
    cub.append({"c": [W / 2, -t / 2, wall_h / 2], "s": [W + 2 * t, t, wall_h], "yaw": 0.0, "col": [206, 206, 201]})
    cub.append({"c": [W / 2, H + t / 2, wall_h / 2], "s": [W + 2 * t, t, wall_h], "yaw": 0.0, "col": [198, 198, 194]})
    cub.append({"c": [-t / 2, H / 2, wall_h / 2], "s": [t, H + 2 * t, wall_h], "yaw": 0.0, "col": [192, 192, 188]})
    cub.append({"c": [W + t / 2, H / 2, wall_h / 2], "s": [t, H + 2 * t, wall_h], "yaw": 0.0, "col": [200, 200, 196]})

    def brown():
        base = 140 + int(rng.integers(0, 45))
        return [base, int(base * 0.72), int(base * 0.42)]

    def rack(cx, y0, y1):
        ln = y1 - y0
        for px in (cx - 0.12, cx + 0.12):
            for py in (y0 + 0.06, y1 - 0.06):
                cub.append({"c": [px, py, 0.55], "s": [0.05, 0.05, 1.1], "yaw": 0.0, "col": [48, 48, 52]})
        for z in (0.42, 0.84):
            cub.append({"c": [cx, (y0 + y1) / 2, z], "s": [0.26, ln - 0.08, 0.05], "yaw": 0.0, "col": [228, 196, 10]})
        for z in (0.62, 1.04):
            n = 3
            for i in range(n):
                cy = y0 + ln * (i + 0.5) / n
                cub.append({"c": [cx, cy, z], "s": [0.28, ln / n * 0.8, 0.34], "yaw": 0.0, "col": brown()})
        if rng.random() < 0.8:
            cub.append({"c": [cx, (y0 + y1) / 2 + float(rng.uniform(-0.3, 0.3)), 1.42],
                        "s": [0.26, 0.4, 0.3], "yaw": 0.0, "col": brown()})
        for i in range(2):
            cy = y0 + ln * (0.25 + 0.5 * i)
            cub.append({"c": [cx, cy, 0.16], "s": [0.24, 0.3, 0.32], "yaw": 0.0, "col": brown()})

    if w.scenario == "corridor":
        x = 0.6
        while x < W - 0.5:
            for yy in (0.55, H - 0.55):
                cub.append({"c": [x, yy, 0.25], "s": [0.4, 0.4, 0.5], "yaw": 0.0, "col": brown()})
                if rng.random() < 0.6:
                    cub.append({"c": [x + 0.05, yy, 0.62], "s": [0.32, 0.32, 0.3], "yaw": 0.0, "col": brown()})
            x += 0.9
    else:
        for c in (15, 35, 55):
            rack(c * 0.1, 0.8, 2.3)
            rack(c * 0.1, 2.8, 4.3)
        cub.append({"c": [3.5, 4.05, 0.28], "s": [0.36, 0.5, 0.56], "yaw": 0.0, "col": [25, 125, 35]})
        cub.append({"c": [3.5, 4.05, 0.60], "s": [0.40, 0.54, 0.08], "yaw": 0.0, "col": [18, 95, 28]})

    for yy in (0.72, 2.5, 4.45):
        if yy < H - 0.2:
            marks.append({"pts": [[0.15, yy - 0.025], [W - 0.15, yy - 0.025],
                                  [W - 0.15, yy + 0.025], [0.15, yy + 0.025]],
                          "col": [0, 150, 70], "z": 0.006})

    x = 0.6
    while x < 3.2:
        col = [230, 200, 0] if int(x / 0.2) % 2 == 0 else [25, 25, 25]
        marks.append({"pts": [[x, 2.94], [x + 0.2, 2.94], [x + 0.2, 3.06], [x, 3.06]], "col": col, "z": 0.006})
        x += 0.2

    y = 0.6
    while y < 2.4:
        col = [230, 200, 0] if int(y / 0.2) % 2 == 0 else [25, 25, 25]
        marks.append({"pts": [[1.94, y], [2.06, y], [2.06, y + 0.2], [1.94, y + 0.2]], "col": col, "z": 0.006})
        y += 0.2

    return {
        "W": r3(W), "H": r3(H), "res": w.res,
        "cols": w.cols, "rows": w.rows,
        "grid": w.grid.reshape(-1).astype(int).tolist(),
        "cuboids": cub,
        "marks": marks,
        "goals": [{"name": g.name, "x": r3(g.pose.x), "y": r3(g.pose.y)} for g in w.goal_list()],
    }


# ----------------------------------------------------------------------
# Video writer (top-down recorder MP4)
# ----------------------------------------------------------------------

def create_video_writer(base_path: Path, fps: int, size: Tuple[int, int]):
    base_path = Path(base_path)
    candidates = [(base_path.with_suffix(".mp4"), "mp4v"), (base_path.with_suffix(".avi"), "XVID")]

    for path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(size[0]), int(size[1])))
        if writer.isOpened():
            return writer, path

    return None, candidates[0][0]


# ----------------------------------------------------------------------
# Embedded Gazebo-style 3D live viewer (HTML/JS)
# ----------------------------------------------------------------------

def make_live_html(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))

    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html,body{margin:0;background:#3f3f3f;font-family:Ubuntu,Arial,sans-serif;color:#ddd;overflow:hidden}
  #menubar{height:26px;background:#4c4c4c;border-bottom:1px solid #333;display:flex;align-items:center;padding:0 10px;font-size:13px}
  #menubar span{margin-right:18px;cursor:default}
  #menubar span u{text-decoration:underline}
  #toolbar{height:34px;background:#5a5a5a;border-bottom:1px solid #333;display:flex;align-items:center;padding:0 8px;gap:6px}
  .ticon{width:22px;height:22px;background:#6a6a6a;border:1px solid #474747;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:12px;color:#cfcfcf}
  #stage{position:relative;background:#8f8f8f}
  canvas{display:block}
  #statusbar{height:30px;background:#5a5a5a;border-top:1px solid #333;display:flex;align-items:center;padding:0 10px;font-size:12px;gap:14px;color:#e8e8e8}
  #statusbar .fld{background:#4a4a4a;border:1px solid #3a3a3a;padding:2px 8px;border-radius:3px}
  #statusbar button{background:#6a6a6a;border:1px solid #474747;color:#eee;border-radius:3px;padding:2px 10px;cursor:pointer}
  #controls{height:34px;background:#4c4c4c;display:flex;align-items:center;gap:8px;padding:0 10px;font-size:12px}
  #controls button,#controls select{background:#6a6a6a;border:1px solid #474747;color:#eee;border-radius:3px;padding:2px 8px;cursor:pointer}
  #controls input[type=range]{flex:1}
</style>
</head>
<body>
<div id="menubar">
  <span><u>F</u>ile</span><span><u>E</u>dit</span><span><u>C</u>amera</span>
  <span><u>V</u>iew</span><span><u>W</u>indow</span><span><u>H</u>elp</span>
</div>
<div id="toolbar">
  <div class="ticon">&#9650;</div><div class="ticon">&#10010;</div><div class="ticon">&#8635;</div>
  <div class="ticon">&#9645;</div><div class="ticon">&#9679;</div><div class="ticon">&#9646;</div>
  <div class="ticon">&#9728;</div><div class="ticon">&#10022;</div><div class="ticon">&#9629;</div>
  <div class="ticon">&#9630;</div><div class="ticon">&#8644;</div><div class="ticon">&#9711;</div>
</div>
<div id="stage"><canvas id="cv" width="1280" height="620"></canvas></div>
<div id="statusbar">
  <button id="gzpause">&#10074;&#10074;</button><button id="gzstep">&#10095;&#10095;</button>
  <span>Steps: <span class="fld" id="steps">1</span></span>
  <span>Real Time Factor: <span class="fld">0.86</span></span>
  <span>Sim Time: <span class="fld" id="simtime">00 00:00:00.000</span></span>
  <span>Real Time: <span class="fld" id="realtime">00 00:00:00.000</span></span>
  <span>Iterations: <span class="fld" id="iters">56538</span></span>
  <span>FPS: <span class="fld" id="fps">0</span></span>
  <button id="gzreset">Reset Time</button>
</div>
<div id="controls">
  <button id="play">Pause</button><button id="restart">Restart</button>
  <span>Speed</span>
  <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
  <span>Camera</span>
  <select id="cam"><option value="fixed" selected>Fixed overlook (screenshot)</option><option value="chase">Chase robot</option></select>
  <label><input type="checkbox" id="minimap"> Minimap</label>
  <input id="timeline" type="range" min="0" max="0" value="0">
  <span id="time">0.0s</span>
</div>

<script>
const DATA = __DATA__;
const S = DATA.scene, FR = DATA.frames;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const W = cv.width, H = cv.height, CX = W/2, CY = H/2 + 30, NEAR = 0.06;
const focal = H * 1.15;
const Ld = norm3([0.35,-0.45,0.82]);

let cam=[0,0,0], R=[1,0,0], U=[0,1,0], F=[0,0,1];
let camPos=[S.W*0.10,-S.H*0.38,Math.min(2.5,S.H*0.52)];
let camMode="fixed";

function norm3(v){const l=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/l,v[1]/l,v[2]/l];}
function sub3(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross3(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}

function setCam(pos,tgt){
  cam=pos;
  F=norm3(sub3(tgt,pos));
  R=norm3(cross3(F,[0,0,1]));
  U=cross3(R,F);
}
function proj(p){
  const d=sub3(p,cam);
  const xc=dot3(d,R), yc=dot3(d,U), zc=dot3(d,F);
  if(zc<NEAR) return null;
  const s=focal/zc;
  return [CX+xc*s, CY-yc*s, zc];
}
function rotYaw(x,y,yaw){const c=Math.cos(yaw),s=Math.sin(yaw);return [x*c-y*s, x*s+y*c];}

let faces=[];
function pushCub(c,s,yaw,col){
  const hx=s[0]/2, hy=s[1]/2, hz=s[2]/2;
  const L=[[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
           [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]];
  const FS=[[[0,1,2,3],[0,0,-1]],[[4,5,6,7],[0,0,1]],[[0,1,5,4],[0,-1,0]],
            [[2,3,7,6],[0,1,0]],[[1,2,6,5],[1,0,0]],[[0,3,7,4],[-1,0,0]]];
  const wp=[];
  for(let i=0;i<8;i++){
    const r=rotYaw(L[i][0],L[i][1],yaw);
    wp.push([c[0]+r[0], c[1]+r[1], c[2]+L[i][2]]);
  }
  for(let f=0;f<6;f++){
    const idx=FS[f][0];
    let nl=FS[f][1];
    const rr=rotYaw(nl[0],nl[1],yaw);
    const nw=[rr[0],rr[1],nl[2]];
    let cx=0,cy=0,cz=0;
    for(const i of idx){cx+=wp[i][0];cy+=wp[i][1];cz+=wp[i][2];}
    cx/=4;cy/=4;cz/=4;
    if(dot3(nw,[cx-cam[0],cy-cam[1],cz-cam[2]])>=0) continue;
    const pts=[];
    let zsum=0, ok=true;
    for(const i of idx){
      const p=proj(wp[i]);
      if(!p){ok=false;break;}
      pts.push([p[0],p[1]]);
      zsum+=p[2];
    }
    if(!ok) continue;
    const li=0.52+0.48*Math.max(0,dot3(nw,Ld));
    faces.push({p:pts, z:zsum/4,
      col:"rgb("+Math.min(255,col[0]*li|0)+","+Math.min(255,col[1]*li|0)+","+Math.min(255,col[2]*li|0)+")"});
  }
}
function drawFaces(){
  faces.sort((a,b)=>b.z-a.z);
  for(const f of faces){
    ctx.beginPath();
    ctx.moveTo(f.p[0][0],f.p[0][1]);
    for(let i=1;i<f.p.length;i++) ctx.lineTo(f.p[i][0],f.p[i][1]);
    ctx.closePath();
    ctx.fillStyle=f.col;
    ctx.fill();
    ctx.strokeStyle=f.col;
    ctx.lineWidth=0.6;
    ctx.stroke();
  }
  faces=[];
}
function poly(pts3,col,alpha){
  const pp=[];
  for(const p of pts3){const q=proj(p);if(!q)return;pp.push(q);}
  ctx.beginPath();
  ctx.moveTo(pp[0][0],pp[0][1]);
  for(let i=1;i<pp.length;i++) ctx.lineTo(pp[i][0],pp[i][1]);
  ctx.closePath();
  ctx.globalAlpha=alpha==null?1:alpha;
  ctx.fillStyle=col;
  ctx.fill();
  ctx.globalAlpha=1;
}
function line3(a,b,col,wd,alpha){
  const p=proj(a),q=proj(b);
  if(!p||!q)return;
  ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);
  ctx.strokeStyle=col;ctx.lineWidth=wd||1;
  ctx.globalAlpha=alpha==null?1:alpha;
  ctx.stroke();ctx.globalAlpha=1;
}

function fmtT(sec){
  const ms=Math.floor((sec%1)*1000);
  const s=Math.floor(sec)%60;
  const m=Math.floor(sec/60)%60;
  return "00 00:"+String(m).padStart(2,"0")+":"+String(s).padStart(2,"0")+"."+String(ms).padStart(3,"0");
}

let idx=0, playing=true, lastTs=0, acc=0, fpsSm=0, wall0=performance.now();
const frameDur=1000/DATA.fps;

function renderFrame(i){
  const f=FR[i], m=f.metrics||{};
  const rp=f.true;

  if(camMode==="chase"){
    const dx=Math.cos(rp[2]), dy=Math.sin(rp[2]);
    const want=[rp[0]-1.9*dx, rp[1]-1.9*dy, 1.25];
    camPos=[camPos[0]+(want[0]-camPos[0])*0.3, camPos[1]+(want[1]-camPos[1])*0.3, camPos[2]+(want[2]-camPos[2])*0.3];
    setCam(camPos,[rp[0]+1.1*dx, rp[1]+1.1*dy, 0.3]);
  } else {
    camPos=[S.W*0.10,-S.H*0.38,Math.min(2.5,S.H*0.52)];
    setCam(camPos,[S.W*0.55,S.H*0.52,0.25]);
  }

  ctx.fillStyle="#9c9c9c";
  ctx.fillRect(0,0,W,H);

  // floor
  poly([[0,0,0],[S.W,0,0],[S.W,S.H,0],[0,S.H,0]],"#b4b4b2",1);

  // floor marks
  for(const mk of S.marks){
    const pts=mk.pts.map(p=>[p[0],p[1],mk.z]);
    poly(pts,"rgb("+mk.col[0]+","+mk.col[1]+","+mk.col[2]+")",1);
  }

  // lidar fan (blue, like Gazebo scan visualization)
  if(f.rays && f.rays.length>2){
    const outer=[[rp[0],rp[1],0.02]];
    const inner=[[rp[0],rp[1],0.025]];
    for(const r of f.rays){
      outer.push([r[0],r[1],0.02]);
      inner.push([rp[0]+(r[0]-rp[0])*0.45, rp[1]+(r[1]-rp[1])*0.45, 0.025]);
    }
    poly(outer,"rgb(70,80,250)",0.26);
    poly(inner,"rgb(60,70,255)",0.40);
    for(const r of f.rays){
      line3([rp[0],rp[1],0.03],[r[0],r[1],0.03],"rgb(150,160,255)",0.7,0.30);
      if(r[2]===1) line3([rp[0],rp[1],0.04],[r[0],r[1],0.04],"rgb(255,255,255)",0.8,0.45);
    }
  }

  // global path on floor
  if(f.path && f.path.length>1){
    for(let k=0;k<f.path.length-1;k++){
      line3([f.path[k][0],f.path[k][1],0.03],[f.path[k+1][0],f.path[k+1][1],0.03],"rgb(255,150,0)",2.2,0.9);
    }
  }

  // goal crosses
  const cur=m.goal||"-";
  for(const g of S.goals){
    const col=g.name===cur?"rgb(0,255,255)":"rgb(180,180,0)";
    line3([g.x-0.14,g.y,0.03],[g.x+0.14,g.y,0.03],col,2.4,1);
    line3([g.x,g.y-0.14,0.03],[g.x,g.y+0.14,0.03],col,2.4,1);
  }

  // static warehouse geometry
  for(const cb of S.cuboids) pushCub(cb.c,cb.s,cb.yaw,cb.col);

  // dynamic obstacles: pallet + carton + white top plate
  for(const o of (f.dyn||[])){
    pushCub([o.x,o.y,0.04],[0.55,0.45,0.06],0,[110,80,40]);
    pushCub([o.x,o.y,0.30],[0.50,0.42,0.45],0,[150,110,60]);
    pushCub([o.x,o.y,0.55],[0.30,0.30,0.04],0,[238,238,232]);
  }
  for(const o of (f.block||[])){
    pushCub([o.x,o.y,0.22],[0.45,0.45,0.44],0,[140,100,55]);
  }

  // robot: grey base + white top plate + lidar puck
  pushCub([rp[0],rp[1],0.09],[0.34,0.30,0.14],rp[2],[120,120,126]);
  pushCub([rp[0],rp[1],0.20],[0.50,0.42,0.07],rp[2],[242,242,238]);
  pushCub([rp[0],rp[1],0.27],[0.07,0.07,0.08],rp[2],[60,60,64]);

  drawFaces();

  // HUD overlay
  ctx.fillStyle="rgba(20,20,20,0.55)";
  ctx.fillRect(10,36,300,86);
  ctx.fillStyle="#eaeaea";
  ctx.font="12px monospace";
  ctx.fillText("STATE: "+f.state+"   goal: "+(m.goal||"-"),18,54);
  ctx.fillText("loc err "+Number(m.err||0).toFixed(3)+" m   ESS "+Number(m.ess||0).toFixed(2),18,70);
  ctx.fillText("lidar "+(Number(m.lidar||0)*100).toFixed(0)+"%   cam "+(Number(m.cam||0)*100).toFixed(0)+"%",18,86);
  ctx.fillText("collisions "+(m.collisions||0)+"   recoveries "+(m.recoveries||0)+"   stops "+(m.safety||0),18,102);
  ctx.fillText("travel "+Number(m.travel||0).toFixed(2)+" m   completed "+(m.completed||0),18,118);

  // minimap
  if(document.getElementById("minimap").checked){
    const mw=170, mh=Math.round(mw*S.H/S.W), mx=W-mw-12, my=44;
    ctx.fillStyle="rgba(15,15,15,0.75)";
    ctx.fillRect(mx-4,my-4,mw+8,mh+8);
    const sc=mw/S.W;
    ctx.fillStyle="#c8c8c8";
    for(let r=0;r<S.rows;r++)for(let c=0;c<S.cols;c++){
      if(S.grid[r*S.cols+c]===1) ctx.fillRect(mx+c*sc,my+r*sc,Math.ceil(sc),Math.ceil(sc));
    }
    if(f.path&&f.path.length>1){
      ctx.beginPath();
      ctx.moveTo(mx+f.path[0][0]*sc,my+f.path[0][1]*sc);
      for(const p of f.path) ctx.lineTo(mx+p[0]*sc,my+p[1]*sc);
      ctx.strokeStyle="#ffb000";ctx.lineWidth=1.4;ctx.stroke();
    }
    ctx.fillStyle="#00ff00";
    ctx.beginPath();ctx.arc(mx+rp[0]*sc,my+rp[1]*sc,3,0,7);ctx.fill();
  }

  // status bar
  document.getElementById("steps").textContent=String(i+1);
  document.getElementById("simtime").textContent=fmtT(f.t||0);
  document.getElementById("realtime").textContent=fmtT((performance.now()-wall0)/1000);
  document.getElementById("iters").textContent=String(56538+i*571);
}

const playBtn=document.getElementById("play");
const timeline=document.getElementById("timeline");
timeline.max=String(FR.length-1);

function setIdx(v){
  idx=Math.max(0,Math.min(FR.length-1,v));
  timeline.value=String(idx);
  document.getElementById("time").textContent=Number(FR[idx].t||0).toFixed(1)+"s";
  renderFrame(idx);
}
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?"Pause":"Play";lastTs=0;acc=0;};
document.getElementById("gzpause").onclick=()=>{playing=false;playBtn.textContent="Play";};
document.getElementById("gzstep").onclick=()=>{playing=false;playBtn.textContent="Play";setIdx(idx+1);};
document.getElementById("gzreset").onclick=()=>{wall0=performance.now();setIdx(0);};
document.getElementById("restart").onclick=()=>{wall0=performance.now();setIdx(0);playing=true;playBtn.textContent="Pause";};
document.getElementById("cam").onchange=e=>{camMode=e.target.value;camPos=[S.W*0.10,-S.H*0.38,Math.min(2.5,S.H*0.52)];setIdx(idx);};
timeline.oninput=e=>setIdx(parseInt(e.target.value,10));

function loop(ts){
  if(playing){
    if(!lastTs)lastTs=ts;
    const dt=ts-lastTs;lastTs=ts;
    if(dt>0)fpsSm=fpsSm*0.9+(1000/dt)*0.1;
    document.getElementById("fps").textContent=fpsSm.toFixed(2);
    acc+=dt*parseFloat(document.getElementById("speed").value);
    while(acc>=frameDur){idx=(idx+1)%FR.length;acc-=frameDur;}
    setIdx(idx);
  }
  requestAnimationFrame(loop);
}
setIdx(0);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""
    return html.replace("__DATA__", payload)


# ----------------------------------------------------------------------
# Demo recording: 3D live payload + top-down MP4
# ----------------------------------------------------------------------

def record_demo(params: Dict[str, Any]) -> Dict[str, Any]:
    sim = NavigationSim(params)
    sim.nav.set_goals(sim.world.goal_list()[:params["demo_goals"]])

    fps = int(params["fps"])
    total_frames = max(30, int(fps * float(params["video_seconds"])))

    out_dir = Path("static")
    out_dir.mkdir(exist_ok=True)
    base_path = out_dir / "live_navigation_demo"

    writer, out_path = create_video_writer(base_path, fps, (CANVAS_W, CANVAS_H))

    frame_records: List[Dict[str, Any]] = []

    for _ in range(total_frames):
        if sim.nav.current_goal is None and len(sim.nav.goal_queue) == 0:
            sim.nav.goal_queue.extend(sim.world.goal_list()[:params["demo_goals"]])
            sim.nav.state = "PLANNING"

        frame = sim.step_once(render=True)
        if frame is not None and writer is not None:
            writer.write(frame)

        frame_records.append(sim.export_frame())

    if writer is not None:
        writer.release()

    video_bytes = b""
    mime = "video/mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        video_bytes = out_path.read_bytes()
        mime = "video/mp4" if out_path.suffix.lower() == ".mp4" else "video/x-msvideo"

    summary = sim.summary()
    summary["scenario"] = params.get("scenario_label", sim.world.scenario)
    summary["mode"] = params.get("mode_label", sim.mode)

    payload = {
        "fps": fps,
        "scene": build_scene(sim),
        "frames": frame_records,
        "summary": summary,
    }

    return {
        "html": make_live_html(payload),
        "video_bytes": video_bytes,
        "mime": mime,
        "path": str(out_path),
        "summary": summary,
        "sim": sim,
    }


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------

def evaluate_scenario(params: Dict[str, Any], episodes: int = 5, max_steps: int = 170) -> pd.DataFrame:
    rows = []
    base_seed = int(params["seed"])

    for ep in range(episodes):
        p = params.copy()
        p["seed"] = base_seed + ep * 17
        p["dynamic_speed"] = max(0.0, float(params["dynamic_speed"]) * (0.75 + 0.5 * ((ep % 3) / 2.0)))

        sim = NavigationSim(p)
        goals = sim.world.goal_list()[:2]
        sim.nav.set_goals(goals)

        success = False
        for _ in range(max_steps):
            sim.step_once(render=False)
            if len(sim.nav.completed) >= len(goals):
                success = True
                break
            if sim.collisions > 0:
                break

        errors = [row["error_m"] for row in sim.telemetry]
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

        rows.append({
            "Episode": ep + 1,
            "Scenario": p.get("scenario_label", p["scenario"]),
            "Mode": p.get("mode_label", p["mode"]),
            "Success": success,
            "Collision": sim.collisions > 0,
            "Time to goal (s)": round(sim.t, 2),
            "Path length (m)": round(sim.travel_distance, 2),
            "Localization RMSE (m)": round(rmse, 3),
            "Recoveries": sim.nav.recoveries,
            "Safety stops": sim.nav.safety_stops,
        })

    return pd.DataFrame(rows)


def compare_localization(params: Dict[str, Any], steps: int = 90) -> pd.DataFrame:
    rows = []

    for mode in ["amcl", "slam", "dead"]:
        p = params.copy()
        p["mode"] = mode
        p["mode_label"] = MODE_LABELS[mode]

        sim = NavigationSim(p)
        sim.nav.set_goals([sim.world.goal_list()[0]])

        for _ in range(steps):
            sim.step_once(render=False)

        errors = [row["error_m"] for row in sim.telemetry]
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        final_error = float(errors[-1]) if errors else 0.0
        mean_cov = float(np.mean([row["cov"] for row in sim.telemetry])) if sim.telemetry else 0.0

        rows.append({
            "Mode": MODE_LABELS[mode],
            "RMSE (m)": round(rmse, 3),
            "Final error (m)": round(final_error, 3),
            "Mean covariance": round(mean_cov, 3),
            "Collisions": sim.collisions,
            "Recoveries": sim.nav.recoveries,
            "Safety stops": sim.nav.safety_stops,
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Streamlit application
# ----------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Autonomous Mobile Robot Navigation Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Autonomous Mobile Robot Navigation Platform")
    st.caption(
        "ROS 2-inspired architecture • Gazebo-style 3D warehouse live simulation embedded in the app • "
        "no physical hardware, no API keys."
    )

    with st.sidebar:
        st.header("Mission Controls")

        scenario_label = st.selectbox("Simulation scenario",
                                      ["Warehouse", "Corridor", "Dynamic Warehouse"], index=0)
        mode_label = st.radio("Localization / mapping mode",
                              ["AMCL (known map)", "Occupancy SLAM (unknown map)", "Dead reckoning"], index=0)

        particles = st.slider("Particle filter size", 30, 180, 70, 10)
        lidar_noise = st.slider("Lidar noise (m)", 0.0, 0.12, 0.035, 0.005)
        dynamic_speed = st.slider("Dynamic obstacle speed (m/s)", 0.0, 1.5, 0.45, 0.05)
        max_vel = st.slider("Max robot speed (m/s)", 0.2, 1.2, 0.70, 0.05)
        safety_margin = st.slider("Safety margin (m)", 0.05, 0.45, 0.15, 0.01)
        video_seconds = st.slider("Live demo length (s)", 6, 20, 10, 1)

        st.divider()

        c1, c2 = st.columns(2)
        step_btn = c1.button("Step", use_container_width=True)
        run10_btn = c2.button("Run 10", use_container_width=True)

        reset_btn = st.button("Reset interactive simulation", use_container_width=True)
        regen_btn = st.button("Regenerate live demo", use_container_width=True)

        st.divider()
        st.markdown(
            "- The **Live Demo** tab shows a Gazebo-style 3D warehouse view that auto-starts.\n"
            "- A top-down MP4 recorder video is also generated and downloadable.\n"
            "- Simulation and hardware interfaces stay separate."
        )

    scenario_map = {"Warehouse": "warehouse", "Corridor": "corridor", "Dynamic Warehouse": "dynamic"}
    mode_map = {"AMCL (known map)": "amcl", "Occupancy SLAM (unknown map)": "slam", "Dead reckoning": "dead"}

    params: Dict[str, Any] = {
        "scenario": scenario_map[scenario_label],
        "scenario_label": scenario_label,
        "mode": mode_map[mode_label],
        "mode_label": mode_label,
        "particles": int(particles),
        "lidar_noise": float(lidar_noise),
        "lidar_drop": 0.02,
        "dynamic_speed": float(dynamic_speed),
        "max_vel": float(max_vel),
        "max_acc": 0.90,
        "safety_margin": float(safety_margin),
        "robot_radius": 0.15,
        "min_ess": 0.22,
        "max_cov": 1.25,
        "min_lidar_valid": 0.60,
        "xy_tolerance": 0.22,
        "fps": DEFAULT_FPS,
        "video_seconds": int(video_seconds),
        "seed": 7,
        "odom_noise": 0.02,
        "process_noise": 0.015,
        "demo_goals": 2,
    }

    param_hash = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    if "demo" not in st.session_state:
        with st.spinner("Building 3D warehouse and rendering embedded live simulation..."):
            st.session_state.demo = record_demo(params)
        st.session_state.demo_hash = param_hash
        st.session_state.sim = st.session_state.demo["sim"]
        st.session_state.sim_hash = param_hash

    if regen_btn:
        with st.spinner("Regenerating 3D live demo and recorder video..."):
            st.session_state.demo = record_demo(params)
        st.session_state.demo_hash = param_hash
        st.session_state.sim = st.session_state.demo["sim"]
        st.session_state.sim_hash = param_hash

    if "sim" not in st.session_state or st.session_state.get("sim_hash") != param_hash:
        if st.session_state.get("demo_hash") == param_hash and "demo" in st.session_state:
            sim = st.session_state.demo["sim"]
        else:
            sim = NavigationSim(params)
            sim.nav.set_goals(sim.world.goal_list()[:params["demo_goals"]])
            sim.latest_frame = sim.render()
        st.session_state.sim = sim
        st.session_state.sim_hash = param_hash
    else:
        sim = st.session_state.sim

    if reset_btn:
        sim = NavigationSim(params)
        sim.nav.set_goals(sim.world.goal_list()[:params["demo_goals"]])
        sim.latest_frame = sim.render()
        st.session_state.sim = sim
        st.session_state.sim_hash = param_hash

    if step_btn:
        sim.step_once(render=True)

    if run10_btn:
        for _ in range(10):
            sim.step_once(render=True)

    demo = st.session_state.demo

    tab_demo, tab_dashboard, tab_eval, tab_arch = st.tabs(
        ["🎥 Live 3D Demo", "🧭 Navigation Dashboard", "📊 Evaluation", "🏗️ ROS 2 Architecture"]
    )

    with tab_demo:
        st.subheader("Gazebo-style 3D warehouse — embedded live simulation (auto-starts)")
        components.html(demo["html"], height=760, scrolling=False)

        st.markdown("#### Top-down recorder video (MP4)")
        if demo["video_bytes"]:
            st.video(demo["video_bytes"], format=demo["mime"])
            st.download_button("Download recorder video", data=demo["video_bytes"],
                               file_name=Path(demo["path"]).name, mime=demo["mime"])
        else:
            st.info("Local video encoder unavailable — the embedded 3D live demo above still runs.")

        if st.session_state.get("demo_hash") != param_hash:
            st.warning("Sidebar parameters changed. Click **Regenerate live demo** to re-render with new settings.")

        s = demo["summary"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Navigation state", s["state"])
        m2.metric("Sim time", f"{s['sim_time_s']} s")
        m3.metric("Goals completed", s["goals_completed"])
        m4.metric("Collisions", s["collisions"])
        m5.metric("Mean loc. error", f"{s['mean_loc_error_m']} m")

        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric("Travel distance", f"{s['travel_distance_m']} m")
        m7.metric("Recoveries", s["recoveries"])
        m8.metric("Safety stops", s["safety_stops"])
        m9.metric("Final loc. error", f"{s['final_loc_error_m']} m")
        m10.metric("Demo length", f"{params['video_seconds']} s")

    with tab_dashboard:
        st.subheader("Live dashboard (top-down)")

        if sim.latest_frame is not None:
            st.image(cv2.cvtColor(sim.latest_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("No rendered frame yet. Use Step or Run 10.")

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("#### Mission queue")
            selected_goal = st.selectbox("Select a goal to queue", [g.name for g in sim.world.goal_list()])
            if st.button("Add selected goal to navigation queue"):
                sim.nav.add_goal(Goal(selected_goal, sim.world.goals[selected_goal]))

            st.markdown("#### Current navigation state")
            st.write({
                "State": sim.nav.state,
                "Current goal": sim.nav.current_goal.name if sim.nav.current_goal else "-",
                "Queued goals": len(sim.nav.goal_queue),
                "Completed goals": len(sim.nav.completed),
                "Path points": len(sim.nav.path),
                "Collisions": sim.collisions,
                "Recoveries": sim.nav.recoveries,
                "Safety stops": sim.nav.safety_stops,
                "Travel distance (m)": round(sim.travel_distance, 2),
            })

        with right:
            st.markdown("#### Sensor and localization health")
            st.write({
                "Lidar valid ratio": round(sim.sensor_stats["lidar_valid"], 2),
                "Lidar obstacle points": sim.sensor_stats["obstacle_points"],
                "Camera confidence": round(sim.sensor_stats["camera_conf"], 2),
                "ESS fraction": round(sim.sensor_stats["ess_frac"], 2),
                "Covariance trace": round(sim.sensor_stats["cov_trace"], 3),
                "Estimated pose x (m)": round(sim.estimate.x, 3),
                "Estimated pose y (m)": round(sim.estimate.y, 3),
                "True pose x (m)": round(sim.robot.true_pose.x, 3),
                "True pose y (m)": round(sim.robot.true_pose.y, 3),
            })

        st.markdown("#### Recent telemetry")
        if sim.telemetry:
            st.dataframe(pd.DataFrame(sim.telemetry[-90:]), use_container_width=True, height=260)
        else:
            st.info("No telemetry recorded yet.")

    with tab_eval:
        st.subheader("Navigation evaluation")

        e1, e2 = st.columns(2)

        if e1.button("Run navigation evaluation (5 episodes)"):
            with st.spinner("Running repeatable navigation episodes..."):
                st.session_state.eval_df = evaluate_scenario(params, episodes=5, max_steps=170)

        if "eval_df" in st.session_state:
            st.dataframe(st.session_state.eval_df, use_container_width=True)
            ev = st.session_state.eval_df
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success rate", f"{ev['Success'].mean() * 100:.1f}%")
            c2.metric("Collision rate", f"{ev['Collision'].mean() * 100:.1f}%")
            c3.metric("Mean path length", f"{ev['Path length (m)'].mean():.2f} m")
            c4.metric("Mean loc. RMSE", f"{ev['Localization RMSE (m)'].mean():.3f} m")

        if e2.button("Compare localization modes"):
            with st.spinner("Comparing AMCL, occupancy SLAM, and dead reckoning..."):
                st.session_state.comp_df = compare_localization(params, steps=90)

        if "comp_df" in st.session_state:
            st.markdown("#### Localization performance comparison")
            st.dataframe(st.session_state.comp_df, use_container_width=True)

    with tab_arch:
        st.subheader("ROS 2-inspired software architecture")
        st.markdown(
            """
            Simulation-first prototype organized like a ROS 2 navigation stack so hardware can be added later.

            **Logical nodes** — `/sim/world_node` (3D warehouse, racks, dynamic obstacles),
            `/robot/state`, `/sensor/lidar`, `/sensor/camera`, `/perception/occupancy_grid`,
            `/localization/particle_filter`, `/planner/global` (A*), `/planner/local` (DWA-style),
            `/control/cmd_vel_limiter`, `/navigator/state_machine`, `/dashboard` (3D live view + top-down),
            `/viz/gazebo_style_viewer` (embedded canvas 3D renderer).

            **Topic-style interfaces** — `/scan`, `/camera/detections`, `/odom`, `/map`, `/localized_pose`,
            `/plan/global`, `/cmd_vel`, `/navigation/state`, `/evaluation/metrics`.

            **Safety behavior** — stop on low lidar validity, low particle ESS, or high covariance;
            local planner rejects unsafe trajectories; recovery behaviors on blockage; collisions counted.

            **Hardware separation** — only simulated sensors/world today; a future hardware adapter can
            publish the same topic-style messages from real devices without touching planner or UI code.
            """
        )


if __name__ == "__main__":
    main()