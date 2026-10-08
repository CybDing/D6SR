"""
Trajectory generators and position tracker for the spherical robot.

Trajectory classes define reference paths with position(t) and velocity(t).
TrajectoryTracker closes the loop: feedforward velocity + PD position
correction → velocity commands for BaselineController.set_drive().

Usage:
    traj = CircleTrajectory(radius=2.0, speed=0.3)
    tracker = TrajectoryTracker(traj)
    vx, vy, ref_x, ref_y = tracker.compute(t, ball_x, ball_y)
    controller.set_drive(vx, vy)
"""

import numpy as np
from abc import ABC, abstractmethod
from control.simple_pid import SimplePID


class Trajectory(ABC):
    """Base class for trajectory generators."""

    @abstractmethod
    def position(self, t: float) -> tuple:
        """Reference position (x, y) at time t."""
        ...

    @abstractmethod
    def velocity(self, t: float) -> tuple:
        """Reference velocity (vx, vy) at time t."""
        ...

    @abstractmethod
    def duration(self) -> float:
        """Total trajectory duration (seconds). inf for periodic."""
        ...


class CircleTrajectory(Trajectory):
    """Circular path at constant speed."""

    def __init__(self, radius=2.0, speed=0.3, center=(0.0, 0.0)):
        self.radius = radius
        self.speed = speed
        self.center = center
        self.omega = speed / radius  # angular velocity (rad/s)

    def position(self, t):
        theta = self.omega * t
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return x, y

    def velocity(self, t):
        theta = self.omega * t
        vx = -self.radius * self.omega * np.sin(theta)
        vy = self.radius * self.omega * np.cos(theta)
        return vx, vy

    def duration(self):
        return float('inf')  # periodic

    def start_position(self):
        """Initial position on the circle (for placing the ball)."""
        return self.position(0.0)


class LineTrajectory(Trajectory):
    """Straight line from start to end at constant speed. Holds at end."""

    def __init__(self, start=(0.0, 0.0), end=(5.0, 0.0), speed=0.3):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.speed = speed
        self._dir = self.end - self.start
        self._length = np.linalg.norm(self._dir)
        self._unit = self._dir / max(self._length, 1e-6)
        self._travel_time = self._length / max(self.speed, 1e-6)

    def position(self, t):
        if t >= self._travel_time:
            return float(self.end[0]), float(self.end[1])
        frac = t / self._travel_time
        pos = self.start + frac * self._dir
        return float(pos[0]), float(pos[1])

    def velocity(self, t):
        if t >= self._travel_time:
            return 0.0, 0.0
        vx = self.speed * self._unit[0]
        vy = self.speed * self._unit[1]
        return float(vx), float(vy)

    def duration(self):
        return self._travel_time


class WaypointTrajectory(Trajectory):
    """Smooth path through a sequence of (x, y) waypoints at constant speed.

    Uses linear interpolation between waypoints. Holds at the last waypoint
    after the path is complete.
    """

    def __init__(self, waypoints, speed=0.5):
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.speed = speed
        # Precompute cumulative distances and segment data
        self._seg_dirs = []
        self._seg_lens = []
        self._cum_dist = [0.0]
        for i in range(len(self.waypoints) - 1):
            d = self.waypoints[i + 1] - self.waypoints[i]
            length = float(np.linalg.norm(d))
            self._seg_lens.append(length)
            self._seg_dirs.append(d / max(length, 1e-6))
            self._cum_dist.append(self._cum_dist[-1] + length)
        self._total_length = self._cum_dist[-1]
        self._total_time = self._total_length / max(self.speed, 1e-6)

    def _segment_at(self, t):
        """Return (segment_index, fraction_in_segment) for time t."""
        dist = min(t * self.speed, self._total_length)
        for i in range(len(self._seg_lens)):
            if dist <= self._cum_dist[i + 1]:
                seg_dist = dist - self._cum_dist[i]
                frac = seg_dist / max(self._seg_lens[i], 1e-6)
                return i, frac
        return len(self._seg_lens) - 1, 1.0

    def position(self, t):
        if t >= self._total_time:
            p = self.waypoints[-1]
            return float(p[0]), float(p[1])
        idx, frac = self._segment_at(t)
        p = self.waypoints[idx] + frac * (self.waypoints[idx + 1] - self.waypoints[idx])
        return float(p[0]), float(p[1])

    def velocity(self, t):
        if t >= self._total_time:
            return 0.0, 0.0
        idx, _ = self._segment_at(t)
        d = self._seg_dirs[idx]
        return float(self.speed * d[0]), float(self.speed * d[1])

    def duration(self):
        return self._total_time


def _arc_points(cx, cy, rx, ry, start_angle, end_angle, n=12):
    """Generate n points along an elliptical arc (angles in radians)."""
    angles = np.linspace(start_angle, end_angle, n)
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]


def _bezier_points(p0, p1, p2, p3, n=12, include_start=True):
    """Sample a cubic Bezier curve with n points."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    if not include_start:
        ts = ts[1:]

    points = []
    for t in ts:
        omt = 1.0 - t
        p = ((omt ** 3) * p0 +
             3.0 * (omt ** 2) * t * p1 +
             3.0 * omt * (t ** 2) * p2 +
             (t ** 3) * p3)
        points.append((float(p[0]), float(p[1])))
    return points


def _connector_points(start, end, n=10):
    """Smooth connector between consecutive characters with gentle arcs."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if np.linalg.norm(end - start) < 1e-6:
        return []

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ctrl_dx = max(abs(dx) * 0.4, 0.15)
    # Bias control points vertically toward the midpoint for a graceful arc
    mid_y = (start[1] + end[1]) * 0.5
    ctrl_dy1 = (mid_y - start[1]) * 0.3
    ctrl_dy2 = (mid_y - end[1]) * 0.3
    p1 = start + np.array([ctrl_dx, ctrl_dy1])
    p2 = end - np.array([ctrl_dx, ctrl_dy2])
    return _bezier_points(start, p1, p2, end, n=n, include_start=False)


# Single-stroke font: each character is a list of (x, y) in normalized [0,1]x[0,1].
# Designed for cursive-style tracing (pen-down path, not outlines).
# Characters are drawn bottom-to-top or left-to-right for natural pen flow.
_PI = np.pi
STROKE_FONT = {
    # I: simple vertical stroke with small serifs
    'I': [(0.3, 0.0), (0.7, 0.0), (0.5, 0.0), (0.5, 1.0),
          (0.3, 1.0), (0.7, 1.0)],

    # R: stem up, top bar, rounded bump, diagonal leg
    'R': ([(0.0, 0.0), (0.0, 1.0), (0.4, 1.0)] +
          list(_arc_points(0.4, 0.75, 0.3, 0.25,
                           _PI/2, -_PI/2, 12)) +
          [(0.0, 0.5), (0.65, 0.0)]),

    # O: full ellipse, starts and ends at top
    'O': list(_arc_points(0.5, 0.5, 0.45, 0.5,
                          _PI/2, _PI/2 + 2*_PI, 24)),

    # S: continuous Bezier "snake", scaled to full [0,1] height, wide arcs.
    'S': (_bezier_points((0.92, 1.00), (0.60, 1.22), (0.05, 1.15),
                         (0.05, 0.72), n=10) +
          _bezier_points((0.05, 0.72), (0.05, 0.52), (0.95, 0.58),
                         (0.95, 0.38), n=10, include_start=False) +
          _bezier_points((0.95, 0.38), (0.95, 0.15), (0.40, -0.12),
                         (0.08, 0.00), n=10, include_start=False)),

    # 2: top arc from upper-left curving right, then diagonal down to baseline
    '2': (list(_arc_points(0.5, 0.72, 0.4, 0.28,
                           _PI*3/4, -_PI/6, 12)) +
          [(0.05, 0.0), (0.95, 0.0)]),

    # 0: full ellipse, same as O
    '0': list(_arc_points(0.5, 0.5, 0.45, 0.5,
                          _PI/2, _PI/2 + 2*_PI, 24)),

    # 6: top hook from upper-right sweeping left-down, then closed bottom circle
    '6': (list(_arc_points(0.55, 0.72, 0.35, 0.28,
                           _PI/4, _PI, 8)) +
          list(_arc_points(0.45, 0.30, 0.35, 0.30,
                           _PI, _PI + 2*_PI, 18))),
}


class TextTrajectory(Trajectory):
    """Trajectory that traces text characters in single-stroke cursive style.

    Each character is drawn as a continuous pen-down path. Adjacent characters
    are joined by a short Bezier ligature to avoid sharp inter-character cuts.

    Usage:
        traj = TextTrajectory("IROS2026", speed=0.5)
    """

    def __init__(self, text, char_width=2.4, char_height=6.0,
                 spacing=0.5, speed=0.5, origin=(0.0, 0.0),
                 connector_samples=10):
        waypoints = []
        cursor_x = origin[0]
        prev_end = None
        for ch in text.upper():
            strokes = STROKE_FONT.get(ch)
            if strokes is None:
                prev_end = None
                cursor_x += char_width * 0.6 + spacing
                continue

            glyph_points = [
                (cursor_x + sx * char_width, origin[1] + sy * char_height)
                for sx, sy in strokes
            ]
            if waypoints and prev_end is not None:
                waypoints.extend(
                    _connector_points(prev_end, glyph_points[0],
                                      n=connector_samples)
                )
            waypoints.extend(glyph_points)
            prev_end = glyph_points[-1]
            cursor_x += char_width + spacing

        if len(waypoints) < 2:
            waypoints = [origin, (origin[0] + 1, origin[1])]

        self._inner = WaypointTrajectory(waypoints, speed=speed)
        self.waypoints = waypoints

    def position(self, t):
        return self._inner.position(t)

    def velocity(self, t):
        return self._inner.velocity(t)

    def duration(self):
        return self._inner.duration()


class FigureEightTrajectory(Trajectory):
    """Figure-eight (lemniscate) path."""

    def __init__(self, radius=1.5, speed=0.2, center=(0.0, 0.0)):
        self.radius = radius
        self.speed = speed
        self.center = center
        self.omega = speed / radius

    def position(self, t):
        wt = self.omega * t
        x = self.center[0] + self.radius * np.sin(wt)
        y = self.center[1] + self.radius * np.sin(wt) * np.cos(wt)
        return x, y

    def velocity(self, t):
        wt = self.omega * t
        w = self.omega
        vx = self.radius * w * np.cos(wt)
        # y = r * sin(wt) * cos(wt) = r/2 * sin(2*wt)
        vy = self.radius * w * (np.cos(wt) * np.cos(wt) - np.sin(wt) * np.sin(wt))
        return vx, vy

    def duration(self):
        return float('inf')  # periodic


# Position PID gains for trajectory tracking
# KP=1.0 balances terrain disturbance rejection vs oscillation (>1.5 oscillates).
# KI=0.05 eliminates steady-state error on sustained slopes within ~10 PID steps.
# KD=0.3 provides moderate damping without amplifying terrain bump noise.
POS_KP = 1.0
POS_KI = 0.05
POS_KD = 0.3
POS_MAX_SPEED = 0.8


class TrajectoryTracker:
    """Position servo: feedforward velocity + PID position correction.

    Computes velocity commands for BaselineController.set_drive().
    """

    def __init__(self, trajectory: Trajectory, dt=0.02, max_speed=POS_MAX_SPEED):
        self.traj = trajectory
        self.max_speed = max_speed
        self.pid_x = SimplePID(POS_KP, ki=POS_KI, kd=POS_KD, dt=dt,
                               output_limit=max_speed)
        self.pid_y = SimplePID(POS_KP, ki=POS_KI, kd=POS_KD, dt=dt,
                               output_limit=max_speed)

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()

    def compute(self, t, ball_x, ball_y):
        """Compute velocity command from current ball position.

        Returns:
            (vx_cmd, vy_cmd, ref_x, ref_y)
        """
        ref_x, ref_y = self.traj.position(t)
        ff_vx, ff_vy = self.traj.velocity(t)

        # PD position correction + feedforward
        vx = ff_vx + self.pid_x.step(ref_x - ball_x)
        vy = ff_vy + self.pid_y.step(ref_y - ball_y)

        # Clamp total speed
        speed = np.hypot(vx, vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale

        return vx, vy, ref_x, ref_y
