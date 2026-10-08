"""
Control allocation helpers for 6-thruster spherical robot.

Wrench order: [Fx, Fy, Fz, Mx, My, Mz] in BODY frame.
Motor order:  [M0, M1, M2, M3, M4, M5]
"""

import numpy as np


def allocation_matrix(r: float = 1.0) -> np.ndarray:
    """Return normalized allocation matrix T (6x6).

    The mapping u -> wrench (body frame) is:
        Fx = M4 + M5
        Fy = M0 + M1
        Fz = M2 + M3
        Mx = r*(M2 - M3)
        My = r*(M4 - M5)
        Mz = r*(M0 - M1)
    """
    return np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # Fx
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Fy
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # Fz
        [0.0, 0.0, r,   -r,  0.0, 0.0],  # Mx
        [0.0, 0.0, 0.0, 0.0, r,   -r ],  # My
        [r,   -r,  0.0, 0.0, 0.0, 0.0],  # Mz
    ], dtype=float)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> rotation matrix (body->world)."""
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),     ww - xx - yy + zz],
    ], dtype=float)


def allocate_wrench(
    force_body: np.ndarray,
    torque_body: np.ndarray,
    motor_radius: float = 0.13,
    force_scale: float = 2.0,
    torque_scale: float = None,
) -> np.ndarray:
    """Allocate body-frame force/torque to 6 motor commands.

    force_scale:  converts desired force to motor fraction (pair sum).
    torque_scale: converts desired torque to motor fraction (pair diff).
    """
    if torque_scale is None:
        torque_scale = 2.0 * motor_radius

    w = np.array([
        force_body[0], force_body[1], force_body[2],
        torque_body[0], torque_body[1], torque_body[2],
    ], dtype=float)

    w[0:3] /= max(force_scale, 1e-9)
    w[3:6] /= max(torque_scale, 1e-9)

    T = allocation_matrix(r=1.0)
    u = np.linalg.pinv(T) @ w
    return u
