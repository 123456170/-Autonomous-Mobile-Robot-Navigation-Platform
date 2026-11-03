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

SCENARIO_LABELS = {
    "warehouse": "Warehouse",
    "corridor": "Corridor",
    "dynamic": "Dynamic Warehouse",
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

        # Shelf blocks creating corridors.
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
            DynamicObstacle(
                "Forklift Alpha",
                1.5,
                2.5,
                0.30,
                [(1.2, 2.5), (5.0, 2.5)],
                0.42,
            ),
            DynamicObstacle(
                "Forklift Beta",
                6.0,
                0.72,
                0.26,
                [(6.2, 0.72), (1.2, 0.72)],
                0.34,
            ),
        ]

        self.blockers = [
            {
                "name": "Temporary pallet",
                "x": 6.2,
                "y": 2.5,
                "radius": 0.42,
                "start": 5.0,
                "end": 12.5,
                "active": False,
            }
        ]

    def _build_corridor(self) -> None:
        self.cols = 90
        self.rows = 32
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self._perimeter()

        # Long indoor corridor with partial pillars.
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
            DynamicObstacle(
                "Shuttle",
                2.0,
                1.60,
                0.26,
                [(2.0, 1.60), (8.5, 1.60)],
                0.45,
            )
        ]

        self.blockers = [
            {
                "name": "Door pallet",
                "x": 5.0,
                "y": 1.60,
                "radius": 0.40,
                "start": 4.0,
                "end": 11.0,
                "active": False,
            }
        ]

    def _build_dynamic_warehouse(self) -> None:
        self._build_warehouse()

        self.dynamic.extend([
            DynamicObstacle(
                "Cart Gamma",
                2.5,
                4.45,
                0.24,
                [(2.5, 4.45), (6.0, 4.45)],
                0.38,
            ),
            DynamicObstacle(
                "Maintenance robot",
                4.5,
                0.72,
                0.22,
                [(4.5, 0.72), (1.5, 0.72), (1.5, 2.5), (4.5, 2.5)],
                0.32,
            ),
        ])

        self.blockers.extend([
            {
                "name": "Falling box",
                "x": 5.0,
                "y": 1.15,
                "radius": 0.38,
                "start": 3.0,
                "end": 9.0,
                "active": False,
            },
            {
                "name": "Cone zone",
                "x": 2.0,
                "y": 4.45,
                "radius": 0.34,
                "start": 10.0,
                "end": 18.0,
                "active": False,
            },
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
                    "x": d.x,
                    "y": d.y,
                    "radius": d.radius,
                    "vx": d.vx,
                    "vy": d.vy,
                    "name": d.name,
                    "kind": "dynamic",
                })
        for b in self.blockers:
            if b.get("active", False):
                out.append({
                    "x": b["x"],
                    "y": b["y"],
                    "radius": b["radius"],
                    "vx": 0.0,
                    "vy": 0.0,
                    "name": b["name"],
                    "kind": "blocker",
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

    def _raycast_with_active(
        self,
        pose: Pose,
        angle: float,
        max_range: float,
        active: List[Dict[str, Any]],
    ) -> float:
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

    def scan(
        self,
        pose: Pose,
        n_rays: int = 60,
        max_range: float = 5.0,
        noise: float = 0.03,
        drop_prob: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def camera_detections(
        self,
        pose: Pose,
        max_range: float = 4.0,
        n_beams: int = 9,
    ) -> List[Dict[str, Any]]:
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
    def __init__(
        self,
        rows: int,
        cols: int,
        res: float,
        known_grid: Optional[np.ndarray] = None,
    ):
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

    def update(
        self,
        pose: Pose,
        ranges: np.ndarray,
        angles: np.ndarray,
        valid: np.ndarray,
        max_range: float,
    ) -> None:
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
                    x = pose.x + ca * d
                    y = pose.y + sa * d
                    self._mark(x, y, False)
                    d += step

                x = pose.x + ca * r
                y = pose.y + sa * r
                self._mark(x, y, True)
            else:
                d = 0.18
                while d < max_range:
                    x = pose.x + ca * d
                    y = pose.y + sa * d
                    self._mark(x, y, False)
                    d += step


# ----------------------------------------------------------------------
# Particle filter localization / mapping support
# ----------------------------------------------------------------------

class ParticleFilter:
    def __init__(
        self,
        n_particles: int,
        initial_pose: Pose,
        spread: float,
        rng: np.random.Generator,
    ):
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

    def _prob_at(
        self,
        x: float,
        y: float,
        map_prob: np.ndarray,
        res: float,
        rows: int,
        cols: int,
    ) -> float:
        c = int(math.floor(x / res))
        r = int(math.floor(y / res))
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 0.95
        return float(map_prob[r, c])

    def sensor_update(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        valid: np.ndarray,
        map_prob: np.ndarray,
        res: float,
        max_range: float,
    ) -> None:
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
                        x = px + ca * r * frac
                        y = py + sa * r * frac
                        p = self._prob_at(x, y, map_prob, res, rows, cols)
                        ray_ll += math.log(max(1.0 - p, 0.08))

                    x = px + ca * r
                    y = py + sa * r
                    p = self._prob_at(x, y, map_prob, res, rows, cols)
                    ray_ll += math.log(max(p, 0.08))
                else:
                    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                        x = px + ca * r * frac
                        y = py + sa * r * frac
                        p = self._prob_at(x, y, map_prob, res, rows, cols)
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
        if math.hypot(cos_sum, sin_sum) < 1e-9:
            theta = float(self.particles[0, 2])
        else:
            theta = math.atan2(sin_sum, cos_sum)
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

def build_costmap(
    world: World,
    occ_prob: np.ndarray,
    obstacles: List[Dict[str, Any]],
    safety_margin: float,
    robot_radius: float,
    mode: str,
    explored: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


def nearest_free_cell(blocked: np.ndarray, rc: Tuple[int, int], max_rad: int = 12) -> Optional[Tuple[int, int]]:
    rows, cols = blocked.shape
    r0, c0 = rc
    r0 = clamp(r0, 0, rows - 1)
    c0 = clamp(c0, 0, cols - 1)

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


def astar(blocked: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    rows, cols = blocked.shape
    if start == goal:
        return [start]

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def heuristic(rc: Tuple[int, int]) -> float:
        return math.hypot(rc[0] - goal[0], rc[1] - goal[1])

    g = np.full((rows, cols), np.inf, dtype=np.float32)
    g[start] = 0.0
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))

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

            step_cost = math.hypot(dr, dc)
            ng = gc + step_cost
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + heuristic((nr, nc)), ng, (nr, nc)))

    return None


def plan_global(
    blocked: np.ndarray,
    world: World,
    start: Pose,
    goal: Pose,
) -> Optional[List[Pose]]:
    start_rc = world.world_to_cell(start.x, start.y)
    goal_rc = world.world_to_cell(goal.x, goal.y)

    start_rc = nearest_free_cell(blocked, start_rc)
    goal_rc = nearest_free_cell(blocked, goal_rc)

    if start_rc is None or goal_rc is None:
        return None

    cells = astar(blocked, start_rc, goal_rc)
    if cells is None:
        return None

    path: List[Pose] = []
    for r, c in cells:
        x, y = world.cell_to_world(r, c)
        path.append(Pose(x, y, 0.0))
    return path


class LocalPlanner:
    def __init__(self, params: Dict[str, Any]):
        self.max_v = float(params["max_vel"])
        self.max_w = self.max_v * 1.8
        self.v_bins = 6
        self.w_bins = 7
        self.horizon_time = 1.0

    def _select_target(self, pose: Pose, path: List[Pose], goal: Goal) -> Tuple[Pose, bool]:
        if not path:
            return goal.pose, True

        target = path[0]
        final = False
        for p in path:
            if dist2d(pose.x, pose.y, p.x, p.y) > 0.40:
                target = p
                final = False
                break
        else:
            target = path[-1]
            final = True

        return target, final

    def _static_clearance(
        self,
        pose: Pose,
        blocked: np.ndarray,
        dist_transform: np.ndarray,
        res: float,
    ) -> float:
        c = int(math.floor(pose.x / res))
        r = int(math.floor(pose.y / res))
        rows, cols = blocked.shape
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 0.0
        if blocked[r, c]:
            return 0.0
        return float(dist_transform[r, c])

    def _dynamic_clearance(
        self,
        pose: Pose,
        obstacles: List[Dict[str, Any]],
        tau: float,
        robot_radius: float,
    ) -> float:
        if not obstacles:
            return 10.0

        min_clear = 10.0
        for obs in obstacles:
            ox = obs["x"] + obs.get("vx", 0.0) * tau
            oy = obs["y"] + obs.get("vy", 0.0) * tau
            clear = dist2d(pose.x, pose.y, ox, oy) - obs["radius"] - robot_radius
            min_clear = min(min_clear, clear)
        return min_clear

    def compute(
        self,
        pose: Pose,
        goal: Goal,
        path: List[Pose],
        blocked: np.ndarray,
        dist_transform: np.ndarray,
        obstacles: List[Dict[str, Any]],
        robot_twist: Twist,
        world_res: float,
        robot_radius: float,
        safety_margin: float,
        dt: float,
    ) -> Tuple[Twist, bool]:
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

                cost = (
                    1.0 * dist_err
                    + 1.2 * heading_err
                    + 0.15 * abs(v - self.max_v * 0.75)
                    + 0.35 * abs(w)
                    - 1.5 * min(min_static, 1.0)
                    - 1.0 * min(min_dyn, 1.0)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_cmd = Twist(float(v), float(w))
                    found = True

        if found:
            return best_cmd, True

        # Fallback: in-place rotation if geometrically safe.
        best_rot_cost = float("inf")
        best_rot_w = 0.0
        found_rot = False

        sd0 = self._static_clearance(pose, blocked, dist_transform, world_res)
        dd0 = self._dynamic_clearance(pose, obstacles, 0.0, robot_radius)

        if sd0 > 0.0 and dd0 > safety_margin * 0.25:
            for w in w_samples:
                heading_err = abs(wrap_pi(target_theta - pose.theta))
                cost = heading_err + 0.2 * abs(w)
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

    def _path_blocked(self, pose: Pose, blocked: np.ndarray, world: World) -> bool:
        if not self.path:
            return True

        for p in self.path[:6]:
            r, c = world.world_to_cell(p.x, p.y)
            if blocked[r, c]:
                return True
        return False

    def compute(
        self,
        est_pose: Pose,
        blocked: np.ndarray,
        dist_transform: np.ndarray,
        world: World,
        obstacles: List[Dict[str, Any]],
        robot_twist: Twist,
        sensor_stats: Dict[str, Any],
        dt: float,
    ) -> Twist:
        self.time += dt

        if self.current_goal is None:
            if len(self.goal_queue) == 0:
                self.state = "ALL_GOALS_COMPLETE" if self.completed else "IDLE"
                return Twist(0.0, 0.0)

            self.current_goal = self.goal_queue.popleft()
            self.path = []
            self.replan_timer = 0.0
            self.state = "PLANNING"

        # Safety monitor: stop if localization or perception is unreliable.
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
            self.completed.append({
                "name": self.current_goal.name,
                "time": self.time,
            })
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
                w = clamp(wrap_pi(target_theta - est_pose.theta), -self.params["max_vel"] * 1.8, self.params["max_vel"] * 1.8)
                return Twist(0.0, w)

            self.path = path
            self.replan_timer = 2.0
            self.state = "FOLLOWING"

        cmd, safe = self.local.compute(
            est_pose,
            self.current_goal,
            self.path,
            blocked,
            dist_transform,
            obstacles,
            robot_twist,
            world.res,
            self.params["robot_radius"],
            self.params["safety_margin"],
            dt,
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
                w = clamp(wrap_pi(target_theta - est_pose.theta), -self.params["max_vel"] * 1.8, self.params["max_vel"] * 1.8)
                return Twist(0.0, w)

            return Twist(0.0, 0.0)

        self.stuck = 0
        if self.state == "RECOVERY":
            self.state = "FOLLOWING"

        return cmd


# ----------------------------------------------------------------------
# Integrated simulation / ROS 2-inspired node orchestration
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
            self.world,
            self.occ_prob,
            self.perceived_obs,
            self.params["safety_margin"],
            self.robot.radius,
            self.mode,
            self.occ.explored if self.mode == "slam" else None,
        )

        self.sensor_stats = {
            "lidar_valid": 1.0,
            "camera_conf": 0.0,
            "ess_frac": 1.0,
            "cov_trace": 0.01,
            "obstacle_points": 0,
        }

        self.telemetry: List[Dict[str, Any]] = []
        self.latest_frame: Optional[np.ndarray] = None

    def step_once(self, render: bool = True) -> Optional[np.ndarray]:
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
                self.robot.true_pose.x,
                self.robot.true_pose.y,
                self.robot.prev_true.x,
                self.robot.prev_true.y,
            )

        ranges, angles, valid = self.world.scan(
            self.robot.true_pose,
            n_rays=60,
            max_range=5.0,
            noise=self.params["lidar_noise"],
            drop_prob=self.params["lidar_drop"],
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
            self.pf.motion_update(
                self.robot.odom_delta,
                noise_xy=self.params["process_noise"],
                noise_th=self.params["process_noise"] * 0.8,
            )

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
            self.world,
            self.occ_prob,
            self.perceived_obs,
            self.params["safety_margin"],
            self.robot.radius,
            self.mode,
            self.occ.explored if self.mode == "slam" else None,
        )

        cmd = self.nav.compute(
            self.estimate,
            self.latest_blocked,
            self.latest_dist,
            self.world,
            self.perceived_obs,
            self.robot.twist,
            self.sensor_stats,
            dt,
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
                rays.append([r3(ex), r3(ey)])

        path_pts = [[r3(p.x), r3(p.y)] for p in self.nav.path]

        dyn: List[Dict[str, Any]] = []
        block: List[Dict[str, Any]] = []
        for obs in self.world.active_obstacles():
            item = {
                "x": r3(obs["x"]),
                "y": r3(obs["y"]),
                "r": r3(obs["radius"]),
                "name": obs["name"],
            }
            if obs["kind"] == "dynamic":
                dyn.append(item)
            else:
                block.append(item)

        err = dist2d(self.estimate.x, self.estimate.y, self.robot.true_pose.x, self.robot.true_pose.y)

        return {
            "t": r3(self.t),
            "state": self.nav.state,
            "true": [r3(self.robot.true_pose.x), r3(self.robot.true_pose.y), r3(self.robot.true_pose.theta)],
            "odom": [r3(self.robot.odom_pose.x), r3(self.robot.odom_pose.y), r3(self.robot.odom_pose.theta)],
            "est": [r3(self.estimate.x), r3(self.estimate.y), r3(self.estimate.theta)],
            "path": path_pts,
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

    def render(self) -> np.ndarray:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        cv2.putText(
            canvas,
            "Autonomous Mobile Robot Navigation — ROS 2-inspired Live Simulation",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Map panel.
        cv2.rectangle(canvas, (12, 48), (735, 610), (65, 65, 65), 1)

        map_x, map_y = 20, 60
        max_w, max_h = 700, 540
        world_w_m = self.world.cols * self.world.res
        world_h_m = self.world.rows * self.world.res
        scale = min(max_w / world_w_m, max_h / world_h_m)

        draw_w = int(world_w_m * scale)
        draw_h = int(world_h_m * scale)

        if self.mode == "slam":
            wall_mask = self.occ_prob > 0.60
        else:
            wall_mask = self.world.grid == 1

        vis = np.zeros((self.world.rows, self.world.cols, 3), dtype=np.uint8)
        vis[:] = (42, 42, 42)
        vis[wall_mask] = (190, 190, 190)
        vis = cv2.resize(vis, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
        canvas[map_y:map_y + draw_h, map_x:map_x + draw_w] = vis

        def w2p(x: float, y: float) -> Tuple[int, int]:
            return int(map_x + x * scale), int(map_y + y * scale)

        # Path.
        if len(self.nav.path) > 1:
            pts = np.array([[w2p(p.x, p.y)] for p in self.nav.path], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 190, 255), 2, cv2.LINE_AA)

        # Goals.
        current_name = self.nav.current_goal.name if self.nav.current_goal else None
        for g in self.world.goal_list():
            pt = w2p(g.pose.x, g.pose.y)
            color = (0, 255, 255) if g.name == current_name else (180, 180, 0)
            cv2.drawMarker(canvas, pt, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.putText(canvas, g.name, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # True active obstacles.
        for obs in self.world.active_obstacles():
            center = w2p(obs["x"], obs["y"])
            radius = max(2, int(obs["radius"] * scale))
            color = (0, 140, 255) if obs["kind"] == "blocker" else (0, 0, 255)
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Perceived obstacles.
        for obs in self.perceived_obs:
            center = w2p(obs["x"], obs["y"])
            radius = max(2, int(obs["radius"] * scale))
            cv2.circle(canvas, center, radius + 2, (0, 255, 0), 1, cv2.LINE_AA)

        # Lidar rays from estimated pose.
        if self.latest_scan is not None and self.latest_angles is not None and self.latest_valid is not None:
            est = self.estimate
            p0 = w2p(est.x, est.y)
            for i in range(0, len(self.latest_scan), 2):
                if not self.latest_valid[i]:
                    continue
                r = float(self.latest_scan[i])
                a = float(self.latest_angles[i])
                ex = est.x + r * math.cos(a)
                ey = est.y + r * math.sin(a)
                p1 = w2p(ex, ey)
                color = (0, 220, 0) if r < 4.75 else (90, 90, 90)
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

        def draw_pose_triangle(img: np.ndarray, pose: Pose, color: Tuple[int, int, int], size: int = 9) -> None:
            x, y = w2p(pose.x, pose.y)
            pts = []
            for ang in [pose.theta, pose.theta + 2.35, pose.theta - 2.35]:
                px = x + size * math.cos(ang)
                py = y + size * math.sin(ang)
                pts.append([int(px), int(py)])
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        draw_pose_triangle(canvas, self.robot.true_pose, (0, 255, 0), 10)
        draw_pose_triangle(canvas, self.estimate, (255, 180, 0), 8)
        draw_pose_triangle(canvas, self.robot.odom_pose, (0, 255, 255), 6)

        # Right dashboard panel.
        panel_x = 755
        cv2.rectangle(canvas, (panel_x - 10, 48), (1270, 685), (70, 70, 70), 1)

        state_colors = {
            "IDLE": (180, 180, 180),
            "PLANNING": (255, 180, 0),
            "FOLLOWING": (0, 220, 0),
            "RECOVERY": (0, 140, 255),
            "SAFETY_STOP": (0, 0, 255),
            "GOAL_REACHED": (255, 255, 0),
            "ALL_GOALS_COMPLETE": (255, 0, 0),
        }
        state_color = state_colors.get(self.nav.state, (220, 220, 220))

        cv2.putText(
            canvas,
            f"STATE: {self.nav.state}",
            (panel_x, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            state_color,
            2,
            cv2.LINE_AA,
        )

        err = dist2d(self.estimate.x, self.estimate.y, self.robot.true_pose.x, self.robot.true_pose.y)
        current_goal_name = self.nav.current_goal.name if self.nav.current_goal else "-"

        lines = [
            f"Scenario: {self.params.get('scenario_label', self.world.scenario)}",
            f"Mode: {self.params.get('mode_label', self.mode)}",
            f"Sim time: {self.t:5.1f}s   Step: {self.step}",
            f"Travel distance: {self.travel_distance:5.2f} m",
            "",
            f"True pose : ({self.robot.true_pose.x:5.2f}, {self.robot.true_pose.y:5.2f}, {math.degrees(self.robot.true_pose.theta):6.1f} deg)",
            f"Odom pose : ({self.robot.odom_pose.x:5.2f}, {self.robot.odom_pose.y:5.2f}, {math.degrees(self.robot.odom_pose.theta):6.1f} deg)",
            f"Est pose  : ({self.estimate.x:5.2f}, {self.estimate.y:5.2f}, {math.degrees(self.estimate.theta):6.1f} deg)",
            f"Loc error : {err:5.3f} m",
            f"ESS frac  : {self.sensor_stats['ess_frac']:5.2f}   Cov trace: {self.sensor_stats['cov_trace']:5.3f}",
            "",
            f"Lidar valid: {self.sensor_stats['lidar_valid'] * 100:5.1f}%   Obstacle pts: {self.sensor_stats['obstacle_points']}",
            f"Camera conf: {self.sensor_stats['camera_conf'] * 100:5.1f}%",
            f"Command    : vx={self.last_cmd.vx:5.2f} m/s   w={self.last_cmd.w:5.2f} rad/s",
            f"Velocity   : vx={self.robot.twist.vx:5.2f} m/s   w={self.robot.twist.w:5.2f} rad/s",
            "",
            f"Current goal : {current_goal_name}",
            f"Queued goals : {len(self.nav.goal_queue)}",
            f"Completed    : {len(self.nav.completed)}",
            f"Path points  : {len(self.nav.path)}",
            f"Collisions   : {self.collisions}",
            f"Recoveries   : {self.nav.recoveries}",
            f"Safety stops : {self.nav.safety_stops}",
        ]

        y = 104
        for line in lines:
            cv2.putText(canvas, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
            y += 20

        def draw_bar(img: np.ndarray, x: int, yy: int, w: int, h: int, frac: float, color: Tuple[int, int, int], label: str) -> None:
            frac = clamp(frac, 0.0, 1.0)
            cv2.rectangle(img, (x, yy), (x + w, yy + h), (95, 95, 95), 1)
            cv2.rectangle(img, (x, yy), (x + int(w * frac), yy + h), color, -1)
            cv2.putText(img, label, (x, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)

        draw_bar(canvas, panel_x, 475, 180, 12, self.sensor_stats["lidar_valid"], (0, 200, 0), "Lidar health")
        draw_bar(canvas, panel_x + 210, 475, 180, 12, self.sensor_stats["camera_conf"], (255, 180, 0), "Camera confidence")
        draw_bar(canvas, panel_x, 515, 180, 12, self.sensor_stats["ess_frac"], (0, 220, 220), "Localization ESS")
        draw_bar(canvas, panel_x + 210, 515, 180, 12, clamp(1.0 - self.sensor_stats["cov_trace"] / 2.0, 0.0, 1.0), (180, 0, 255), "Low uncertainty")

        # Mini camera strip.
        cam_img = np.zeros((100, 190, 3), dtype=np.uint8)
        cv2.rectangle(cam_img, (0, 0), (189, 99), (80, 80, 80), 1)
        for det in self.latest_camera:
            x = int(((det["bearing"] + 0.60) / 1.20) * 189)
            x = clamp(x, 0, 186)
            hh = int(det["confidence"] * 90)
            color = (0, 200, 255) if det["type"] != "free" else (130, 130, 130)
            cv2.rectangle(cam_img, (x, 100 - hh), (x + 2, 100), color, -1)

        canvas[560:660, panel_x:panel_x + 190] = cam_img
        cv2.putText(canvas, "Forward camera detections", (panel_x, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "Legend: green=true robot, blue=estimated pose, yellow=odometry, red=dynamic obstacle, orange=temporary blocker",
            (18, 695),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        return canvas


# ----------------------------------------------------------------------
# Video generation
# ----------------------------------------------------------------------

def create_video_writer(base_path: Path, fps: int, size: Tuple[int, int]):
    base_path = Path(base_path)
    candidates = [
        (base_path.with_suffix(".mp4"), "mp4v"),
        (base_path.with_suffix(".avi"), "XVID"),
    ]

    for path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(size[0]), int(size[1])))
        if writer.isOpened():
            return writer, path

    return None, candidates[0][0]


def build_demo_payload(sim: NavigationSim, frames: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    summary = sim.summary()
    summary["scenario"] = params.get("scenario_label", sim.world.scenario)
    summary["mode"] = params.get("mode_label", sim.mode)

    return {
        "fps": int(params["fps"]),
        "world": {
            "cols": int(sim.world.cols),
            "rows": int(sim.world.rows),
            "res": float(sim.world.res),
            "width_m": r3(sim.world.cols * sim.world.res),
            "height_m": r3(sim.world.rows * sim.world.res),
            "grid": sim.world.grid.reshape(-1).astype(int).tolist(),
            "goals": [
                {
                    "name": g.name,
                    "x": r3(g.pose.x),
                    "y": r3(g.pose.y),
                    "theta": r3(g.pose.theta),
                }
                for g in sim.world.goal_list()
            ],
        },
        "frames": frames,
        "summary": summary,
    }


def make_live_html(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))

    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    margin: 0;
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
  }
  .wrap {
    padding: 10px;
  }
  canvas {
    width: 100%;
    height: auto;
    background: #121212;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 8px;
  }
  button {
    background: #222;
    color: #eee;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 6px 10px;
    cursor: pointer;
  }
  button:hover {
    background: #333;
  }
  select, input[type=range] {
    background: #222;
    color: #eee;
    border: 1px solid #555;
    border-radius: 6px;
  }
  .label {
    font-size: 12px;
    color: #bbb;
  }
  #status {
    font-size: 12px;
    color: #9f9;
    margin-top: 4px;
  }
</style>
</head>
<body>
<div class="wrap">
  <canvas id="cv" width="1280" height="680"></canvas>
  <div class="controls">
    <button id="play">Pause</button>
    <button id="restart">Restart</button>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <span class="label">Speed</span>
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
    </select>
    <input id="timeline" type="range" min="0" max="0" value="0" style="flex:1">
    <span id="time" class="label">0.0s</span>
  </div>
  <div id="status">Loading live simulation demo...</div>
</div>

<script>
const DATA = __DATA__;

(function () {
  const statusEl = document.getElementById("status");

  if (!DATA || !DATA.frames || DATA.frames.length === 0) {
    statusEl.textContent = "No demo frames were generated.";
    return;
  }

  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const W = cv.width;
  const H = cv.height;

  const world = DATA.world;
  const frames = DATA.frames;

  const panelX = 760;
  const mapBox = { x: 18, y: 52, w: 720, h: 545 };

  const scale = Math.min(
    (mapBox.w - 20) / world.width_m,
    (mapBox.h - 20) / world.height_m
  );

  const ox = mapBox.x + 10;
  const oy = mapBox.y + 10;
  const cell = world.res * scale;

  const mapW = Math.ceil(world.cols * cell);
  const mapH = Math.ceil(world.rows * cell);

  const mapCanvas = document.createElement("canvas");
  mapCanvas.width = mapW;
  mapCanvas.height = mapH;
  const mctx = mapCanvas.getContext("2d");

  mctx.fillStyle = "#2a2a2a";
  mctx.fillRect(0, 0, mapW, mapH);

  mctx.fillStyle = "#c8c8c8";
  for (let r = 0; r < world.rows; r++) {
    for (let c = 0; c < world.cols; c++) {
      if (world.grid[r * world.cols + c] === 1) {
        mctx.fillRect(
          Math.floor(c * cell),
          Math.floor(r * cell),
          Math.ceil(cell) + 1,
          Math.ceil(cell) + 1
        );
      }
    }
  }

  function wx(x) {
    return ox + x * scale;
  }

  function wy(y) {
    return oy + y * scale;
  }

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function drawTriangle(pose, color, size) {
    const x = wx(pose[0]);
    const y = wy(pose[1]);
    const th = pose[2];

    ctx.beginPath();
    ctx.moveTo(x + size * Math.cos(th), y + size * Math.sin(th));
    ctx.lineTo(x + size * Math.cos(th + 2.35), y + size * Math.sin(th + 2.35));
    ctx.lineTo(x + size * Math.cos(th - 2.35), y + size * Math.sin(th - 2.35));
    ctx.closePath();

    ctx.fillStyle = color;
    ctx.fill();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawBar(x, y, w, h, frac, color, label) {
    frac = clamp(frac, 0, 1);

    ctx.strokeStyle = "#666";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = color;
    ctx.fillRect(x, y, Math.floor(w * frac), h);

    ctx.fillStyle = "#ddd";
    ctx.font = "12px Arial";
    ctx.fillText(label, x, y - 4);
  }

  const stateColors = {
    IDLE: "#b4b4b4",
    PLANNING: "#ffb000",
    FOLLOWING: "#00dc00",
    RECOVERY: "#008cff",
    SAFETY_STOP: "#ff0000",
    GOAL_REACHED: "#ffff00",
    ALL_GOALS_COMPLETE: "#ff00ff"
  };

  function drawFrame(i) {
    const f = frames[i];
    const m = f.metrics || {};

    ctx.fillStyle = "#121212";
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = "#eee";
    ctx.font = "16px Arial";
    ctx.fillText("Autonomous Mobile Robot Navigation — Embedded Live Simulation", 18, 28);

    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    ctx.strokeRect(12, 44, mapBox.w + 12, mapBox.h + 12);

    ctx.drawImage(mapCanvas, ox, oy);

    // Global path.
    if (f.path && f.path.length > 1) {
      ctx.beginPath();
      ctx.moveTo(wx(f.path[0][0]), wy(f.path[0][1]));
      for (let k = 1; k < f.path.length; k++) {
        ctx.lineTo(wx(f.path[k][0]), wy(f.path[k][1]));
      }
      ctx.strokeStyle = "#ffb000";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Lidar rays.
    if (f.rays && f.rays.length > 0) {
      ctx.strokeStyle = "rgba(0, 220, 0, 0.35)";
      ctx.lineWidth = 1;
      const tp = f.true;
      for (let k = 0; k < f.rays.length; k++) {
        ctx.beginPath();
        ctx.moveTo(wx(tp[0]), wy(tp[1]));
        ctx.lineTo(wx(f.rays[k][0]), wy(f.rays[k][1]));
        ctx.stroke();
      }
    }

    // Goals.
    const currentGoal = m.goal || "-";
    for (let g = 0; g < world.goals.length; g++) {
      const goal = world.goals[g];
      const gx = wx(goal.x);
      const gy = wy(goal.y);
      const col = goal.name === currentGoal ? "#00ffff" : "#b4b400";

      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(gx - 7, gy);
      ctx.lineTo(gx + 7, gy);
      ctx.moveTo(gx, gy - 7);
      ctx.lineTo(gx, gy + 7);
      ctx.stroke();

      ctx.fillStyle = col;
      ctx.font = "11px Arial";
      ctx.fillText(goal.name, gx + 9, gy - 7);
    }

    // Temporary blockers.
    if (f.block) {
      for (let k = 0; k < f.block.length; k++) {
        const o = f.block[k];
        ctx.beginPath();
        ctx.arc(wx(o.x), wy(o.y), Math.max(2, o.r * scale), 0, Math.PI * 2);
        ctx.fillStyle = "#008cff";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Dynamic obstacles.
    if (f.dyn) {
      for (let k = 0; k < f.dyn.length; k++) {
        const o = f.dyn[k];
        ctx.beginPath();
        ctx.arc(wx(o.x), wy(o.y), Math.max(2, o.r * scale), 0, Math.PI * 2);
        ctx.fillStyle = "#ff0000";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(wx(o.x), wy(o.y), Math.max(3, o.r * scale + 2), 0, Math.PI * 2);
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Robot poses.
    drawTriangle(f.est, "#ffb000", 8);
    drawTriangle(f.odom, "#ffff00", 6);
    drawTriangle(f.true, "#00ff00", 10);

    // Dashboard panel.
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.strokeRect(panelX - 10, 44, 500, 600);

    const state = f.state || "IDLE";
    ctx.fillStyle = stateColors[state] || "#dddddd";
    ctx.font = "bold 18px Arial";
    ctx.fillText("STATE: " + state, panelX, 72);

    const lines = [
      "Scenario: " + (DATA.summary.scenario || ""),
      "Mode: " + (DATA.summary.mode || ""),
      "Sim time: " + Number(f.t || 0).toFixed(1) + "s",
      "Goal: " + (m.goal || "-"),
      "Completed: " + Number(m.completed || 0),
      "Loc error: " + Number(m.err || 0).toFixed(3) + " m",
      "ESS frac: " + Number(m.ess || 0).toFixed(2),
      "Cov trace: " + Number(m.cov || 0).toFixed(3),
      "Lidar valid: " + (Number(m.lidar || 0) * 100).toFixed(1) + "%",
      "Camera conf: " + (Number(m.cam || 0) * 100).toFixed(1) + "%",
      "Cmd vx: " + Number(m.vx || 0).toFixed(2) + " m/s",
      "Cmd w: " + Number(m.w || 0).toFixed(2) + " rad/s",
      "Travel: " + Number(m.travel || 0).toFixed(2) + " m",
      "Collisions: " + Number(m.collisions || 0),
      "Recoveries: " + Number(m.recoveries || 0),
      "Safety stops: " + Number(m.safety || 0)
    ];

    ctx.fillStyle = "#eeeeee";
    ctx.font = "13px monospace";
    let y = 100;
    for (let k = 0; k < lines.length; k++) {
      ctx.fillText(lines[k], panelX, y);
      y += 19;
    }

    drawBar(panelX, 440, 180, 12, Number(m.lidar || 0), "#00c800", "Lidar health");
    drawBar(panelX + 210, 440, 180, 12, Number(m.cam || 0), "#ffb000", "Camera confidence");
    drawBar(panelX, 480, 180, 12, Number(m.ess || 0), "#00dcdc", "Localization ESS");
    drawBar(
      panelX + 210,
      480,
      180,
      12,
      clamp(1.0 - Number(m.cov || 0) / 2.0, 0.0, 1.0),
      "#b400ff",
      "Low uncertainty"
    );

    ctx.fillStyle = "#aaaaaa";
    ctx.font = "11px Arial";
    ctx.fillText(
      "Legend: green=true robot, orange=estimate, yellow=odometry, red=dynamic, blue=blocker",
      18,
      635
    );
  }

  let idx = 0;
  let playing = true;
  let lastTs = 0;
  let acc = 0;
  const frameDuration = 1000 / DATA.fps;

  const playBtn = document.getElementById("play");
  const restartBtn = document.getElementById("restart");
  const prevBtn = document.getElementById("prev");
  const nextBtn = document.getElementById("next");
  const speedSel = document.getElementById("speed");
  const timeline = document.getElementById("timeline");
  const timeEl = document.getElementById("time");

  timeline.max = String(frames.length - 1);

  function setIdx(v) {
    idx = clamp(v, 0, frames.length - 1);
    timeline.value = String(idx);
    timeEl.textContent = Number(frames[idx].t || 0).toFixed(1) + "s";
    drawFrame(idx);

    statusEl.textContent =
      "Frame " + idx + " / " + (frames.length - 1) +
      " • state: " + (frames[idx].state || "-") +
      " • goal: " + ((frames[idx].metrics && frames[idx].metrics.goal) || "-");
  }

  playBtn.onclick = function () {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
    lastTs = 0;
    acc = 0;
  };

  restartBtn.onclick = function () {
    setIdx(0);
    playing = true;
    playBtn.textContent = "Pause";
  };

  prevBtn.onclick = function () {
    setIdx(idx - 1);
  };

  nextBtn.onclick = function () {
    setIdx(idx + 1);
  };

  timeline.oninput = function () {
    setIdx(parseInt(this.value, 10));
  };

  function loop(ts) {
    if (playing) {
      if (!lastTs) lastTs = ts;
      const dt = ts - lastTs;
      lastTs = ts;

      acc += dt * parseFloat(speedSel.value);

      while (acc >= frameDuration) {
        idx++;
        if (idx >= frames.length) idx = 0;
        acc -= frameDuration;
      }

      setIdx(idx);
    }

    requestAnimationFrame(loop);
  }

  setIdx(0);
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""

    return html.replace("__DATA__", payload)


def record_demo(params: Dict[str, Any]) -> Dict[str, Any]:
    sim = NavigationSim(params)
    goals = sim.world.goal_list()[:params["demo_goals"]]
    sim.nav.set_goals(goals)

    fps = int(params["fps"])
    total_frames = max(30, int(fps * float(params["video_seconds"])))

    out_dir = Path("static")
    out_dir.mkdir(exist_ok=True)
    base_path = out_dir / "live_navigation_demo"

    writer, out_path = create_video_writer(base_path, fps, (CANVAS_W, CANVAS_H))

    frame_records: List[Dict[str, Any]] = []

    for _ in range(total_frames):
        # Keep the demo active if all queued goals are completed early.
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

    payload = build_demo_payload(sim, frame_records, params)
    html = make_live_html(payload)

    return {
        "html": html,
        "video_bytes": video_bytes,
        "mime": mime,
        "path": str(out_path),
        "summary": sim.summary(),
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
        "ROS 2-inspired architecture, simulation-only deployment, free local stack, "
        "no physical hardware required for the initial implementation."
    )

    with st.sidebar:
        st.header("Mission Controls")

        scenario_label = st.selectbox(
            "Simulation scenario",
            ["Warehouse", "Corridor", "Dynamic Warehouse"],
            index=0,
        )

        mode_label = st.radio(
            "Localization / mapping mode",
            ["AMCL (known map)", "Occupancy SLAM (unknown map)", "Dead reckoning"],
            index=0,
        )

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
        regen_btn = st.button("Regenerate live demo video", use_container_width=True)

        st.divider()
        st.subheader("Notes")
        st.markdown(
            "- The app automatically renders an embedded live simulation demo on first run.\n"
            "- A downloadable MP4/AVI video is also generated when the local encoder is available.\n"
            "- Simulation and hardware interfaces are kept separate.\n"
            "- No API keys or cloud services are used."
        )

    scenario_map = {
        "Warehouse": "warehouse",
        "Corridor": "corridor",
        "Dynamic Warehouse": "dynamic",
    }
    mode_map = {
        "AMCL (known map)": "amcl",
        "Occupancy SLAM (unknown map)": "slam",
        "Dead reckoning": "dead",
    }

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

    param_hash = hashlib.md5(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    # Initial automatic live demo generation.
    if "demo" not in st.session_state:
        with st.spinner("Loading realistic demo data and rendering embedded live simulation..."):
            st.session_state.demo = record_demo(params)
        st.session_state.demo_hash = param_hash
        st.session_state.sim = st.session_state.demo["sim"]
        st.session_state.sim_hash = param_hash

    if regen_btn:
        with st.spinner("Regenerating live navigation demo video and embedded animation..."):
            st.session_state.demo = record_demo(params)
        st.session_state.demo_hash = param_hash
        st.session_state.sim = st.session_state.demo["sim"]
        st.session_state.sim_hash = param_hash

    # Ensure interactive simulation exists and matches current parameters.
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
        ["🎥 Live Demo", "🧭 Navigation Dashboard", "📊 Evaluation", "🏗️ ROS 2 Architecture"]
    )

    with tab_demo:
        st.subheader("Embedded live simulation demo")

        components.html(demo["html"], height=790, scrolling=False)

        st.markdown("#### Generated video file")
        if demo["video_bytes"]:
            st.video(demo["video_bytes"], format=demo["mime"])
            st.download_button(
                "Download live demo video",
                data=demo["video_bytes"],
                file_name=Path(demo["path"]).name,
                mime=demo["mime"],
            )
        else:
            st.info(
                "The embedded live canvas demo above is running. "
                "A local video encoder was unavailable, so no downloadable MP4/AVI file was created."
            )

        if st.session_state.get("demo_hash") != param_hash:
            st.warning(
                "Sidebar parameters changed. The embedded demo and video above were generated with the previous settings. "
                "Click **Regenerate live demo video** to render a new clip."
            )

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
        st.subheader("Live dashboard")

        if sim.latest_frame is not None:
            st.image(cv2.cvtColor(sim.latest_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("No rendered frame available yet. Use Step or Run 10.")

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("#### Mission queue")
            goal_names = [g.name for g in sim.world.goal_list()]
            selected_goal = st.selectbox("Select a goal to queue", goal_names)

            if st.button("Add selected goal to navigation queue"):
                pose = sim.world.goals[selected_goal]
                sim.nav.add_goal(Goal(selected_goal, pose))

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
                "Estimated pose theta (deg)": round(math.degrees(sim.estimate.theta), 1),
                "True pose x (m)": round(sim.robot.true_pose.x, 3),
                "True pose y (m)": round(sim.robot.true_pose.y, 3),
            })

        st.markdown("#### Recent telemetry")
        if sim.telemetry:
            df = pd.DataFrame(sim.telemetry[-90:])
            st.dataframe(df, use_container_width=True, height=260)
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
            This prototype is intentionally simulation-first, but it is organized like a ROS 2 navigation stack
            so that hardware interfaces can be added later without changing the core navigation logic.

            **Logical nodes / modules**

            - `/sim/world_node`: indoor warehouse, corridors, dynamic obstacles, temporary blockers.
            - `/robot/state`: differential-drive robot state, odometry integration, acceleration limits.
            - `/sensor/lidar`: simulated 2D lidar scans with noise and dropped rays.
            - `/sensor/camera`: simulated forward camera detections and confidence values.
            - `/perception/occupancy_grid`: occupancy grid construction and obstacle detection.
            - `/localization/particle_filter`: AMCL-style localization or occupancy-SLAM-style mapping.
            - `/planner/global`: A* global planner on an inflated costmap.
            - `/planner/local`: DWA-style local planner with dynamic obstacle prediction.
            - `/control/cmd_vel_limiter`: velocity, acceleration, and safety constraint enforcement.
            - `/navigator/state_machine`: goal queue, replanning, recovery, safety stop behavior.
            - `/dashboard`: embedded live canvas demo, OpenCV-rendered dashboard, map, robot pose, path,
              obstacles, sensor status, and navigation state.

            **Important topic-style interfaces**

            - `/scan`: lidar ranges, angles, validity.
            - `/camera/detections`: bearing, range, confidence, object type.
            - `/odom`: noisy odometry pose and twist.
            - `/map`: occupancy grid or known map reference.
            - `/localized_pose`: estimated pose and uncertainty.
            - `/plan/global`: global path waypoints.
            - `/cmd_vel`: commanded twist after local planning and safety limiting.
            - `/navigation/state`: IDLE, PLANNING, FOLLOWING, RECOVERY, SAFETY_STOP, GOAL_REACHED, ALL_GOALS_COMPLETE.
            - `/evaluation/metrics`: success, collision, path length, time to goal, localization error, recoveries.

            **Hardware separation**

            The current implementation uses only simulated sensors and a simulated world.
            A future hardware adapter can publish the same topic-style messages from real lidar, camera,
            odometry, and motor controllers without modifying planners, localization, or dashboard code.
            """
        )

        st.markdown("#### Safety behavior")
        st.markdown(
            """
            - The robot stops if lidar validity falls below the configured threshold.
            - The robot stops if particle-filter ESS is too low or pose covariance is too high.
            - The local planner rejects trajectories with insufficient static or dynamic clearance.
            - The navigator enters recovery mode when no safe local command is found.
            - Collision events are reverted in simulation and counted as safety failures.
            """
        )


if __name__ == "__main__":
    main()