"""
Cascaded PID controller — port of STM32 cascaded_pid.c

    angle_error → [Outer PID] → velocity_target → [Inner PID] → ctrl_output

Gravity compensation is a feedforward term added AFTER the inner PID.
It is independent of inner PID gains — set gravity_comp_gain=0 to disable.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CascadedPIDConfig:
    """All angles in degrees, velocities in deg/s, output normalized to [-1,1]."""
    # Outer loop (angle → velocity target)
    outer_kp: float = 0.5
    outer_ki: float = 0.05
    outer_kd: float = 0.01
    outer_max_output: float = 60.0     # deg/s
    outer_deadband: float = 2.0        # degrees

    # Inner loop (velocity → ctrl output)
    inner_kp: float = 0.005
    inner_ki: float = 0.001
    inner_kd: float = 0.001
    inner_max_output: float = 0.8      # ctrl fraction

    dt: float = 0.02                   # 50 Hz
    d_filter_alpha: float = 0.4
    velocity_slew_rate: float = 5.0    # deg/s per step
    output_slew_rate: float = 0.05     # ctrl per step
    outer_integral_limit: float = 100.0
    inner_integral_limit: float = 100.0

    # Gravity compensation (feedforward, independent of PID gains)
    gravity_comp_gain: float = 0.0
    stable_angle_deg: float = 0.0
    grav_sign: int = 1

    use_angle_wrap: bool = True


class CascadedPID:
    def __init__(self, config: CascadedPIDConfig, axis_name: str = ""):
        self.cfg = config
        self.axis_name = axis_name
        self.dbg = {}
        self.reset()

    def reset(self):
        self.outer_integral = 0.0
        self.outer_prev_angle = 0.0
        self.outer_d_filtered = 0.0
        self.inner_integral = 0.0
        self.inner_prev_velocity = 0.0
        self.inner_d_filtered = 0.0
        self.velocity_target = 0.0
        self.prev_velocity_target = 0.0
        self.prev_output = 0.0
        self.target_angle = 0.0
        self.dbg = {}

    def set_target(self, angle_deg: float):
        self.target_angle = angle_deg

    def outer_step(self, current_angle: float) -> float:
        """Run outer loop only: angle → Euler rate target (deg/s).

        Call inner_step() afterwards with the (possibly transformed) velocity
        target and the measured body-frame angular velocity.
        """
        cfg = self.cfg

        if cfg.use_angle_wrap:
            angle_error = self._wrap_error(self.target_angle, current_angle)
        else:
            angle_error = self.target_angle - current_angle

        if abs(angle_error) < cfg.outer_deadband:
            angle_error = 0.0
            self.outer_integral *= 0.95

        outer_p = cfg.outer_kp * angle_error

        self.outer_integral = float(np.clip(
            self.outer_integral + angle_error * cfg.dt,
            -cfg.outer_integral_limit, cfg.outer_integral_limit))
        outer_i = cfg.outer_ki * self.outer_integral

        outer_d_raw = -(current_angle - self.outer_prev_angle) / max(cfg.dt, 1e-3)
        self.outer_d_filtered = (cfg.d_filter_alpha * outer_d_raw
                                 + (1 - cfg.d_filter_alpha) * self.outer_d_filtered)
        outer_d = cfg.outer_kd * self.outer_d_filtered

        vel_raw = float(np.clip(outer_p + outer_i + outer_d,
                                -cfg.outer_max_output, cfg.outer_max_output))

        self.velocity_target = float(np.clip(
            vel_raw,
            self.prev_velocity_target - cfg.velocity_slew_rate,
            self.prev_velocity_target + cfg.velocity_slew_rate))
        self.prev_velocity_target = self.velocity_target
        self.outer_prev_angle = current_angle

        # Store partial debug info for outer loop
        self.dbg = {
            'angle_error': angle_error,
            'outer_p': outer_p, 'outer_i': outer_i, 'outer_d': outer_d,
            'vel_target': self.velocity_target,
        }

        return self.velocity_target

    def inner_step(self, velocity_target: float, current_velocity: float,
                   current_angle: float = 0.0) -> float:
        """Run inner loop only: velocity target → ctrl output.

        Args:
            velocity_target: Desired angular velocity (deg/s), may be the
                transformed body-rate target from euler_rate_to_body_rate().
            current_velocity: Measured body angular velocity (deg/s).
            current_angle: Current Euler angle (deg), used for gravity comp.
        """
        cfg = self.cfg

        velocity_error = velocity_target - current_velocity

        inner_p = cfg.inner_kp * velocity_error

        self.inner_integral = float(np.clip(
            self.inner_integral + velocity_error * cfg.dt,
            -cfg.inner_integral_limit, cfg.inner_integral_limit))
        inner_i = cfg.inner_ki * self.inner_integral

        inner_d_raw = -(current_velocity - self.inner_prev_velocity) / max(cfg.dt, 1e-3)
        self.inner_d_filtered = (cfg.d_filter_alpha * inner_d_raw
                                 + (1 - cfg.d_filter_alpha) * self.inner_d_filtered)
        inner_d = cfg.inner_kd * self.inner_d_filtered

        output_raw = float(np.clip(inner_p + inner_i + inner_d,
                                   -cfg.inner_max_output, cfg.inner_max_output))

        grav_comp = 0.0
        if abs(cfg.gravity_comp_gain) > 1e-6:
            grav_comp = cfg.grav_sign * cfg.gravity_comp_gain * np.sin(
                np.radians(current_angle - cfg.stable_angle_deg))
            output_raw = float(np.clip(output_raw + grav_comp,
                                       -cfg.inner_max_output, cfg.inner_max_output))

        output = float(np.clip(
            output_raw,
            self.prev_output - cfg.output_slew_rate,
            self.prev_output + cfg.output_slew_rate))
        self.prev_output = output
        self.inner_prev_velocity = current_velocity

        # Append inner-loop debug info
        self.dbg.update({
            'vel_target': velocity_target, 'vel_error': velocity_error,
            'inner_p': inner_p, 'inner_i': inner_i, 'inner_d': inner_d,
            'grav_comp': grav_comp, 'output_raw': output_raw, 'output': output,
        })

        return output

    def step(self, current_angle: float, current_velocity: float) -> float:
        """One cascaded PID step. Returns ctrl in [-max_output, +max_output].

        Equivalent to outer_step() + inner_step() without rate transform.
        """
        vel_target = self.outer_step(current_angle)
        return self.inner_step(vel_target, current_velocity, current_angle)

    @staticmethod
    def euler_rate_to_body_rate(euler_rates_deg: np.ndarray,
                                roll_deg: float, pitch_deg: float) -> np.ndarray:
        """Convert Euler angle rates [ṙ, ṗ, ẏ] to body angular rates [ω_x, ω_y, ω_z].

        Uses the kinematic relationship ω = E(φ,θ) × euler_rate where:
            E = [[1,  0,       -sin(θ)      ],
                 [0,  cos(φ),   sin(φ)cos(θ)],
                 [0, -sin(φ),   cos(φ)cos(θ)]]
        φ = roll, θ = pitch.  All inputs/outputs in degrees(/s).
        """
        phi = np.radians(roll_deg)
        theta = np.radians(pitch_deg)
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        E = np.array([
            [1.0,  0.0,  -st       ],
            [0.0,  cp,    sp * ct  ],
            [0.0, -sp,    cp * ct  ],
        ])
        return E @ euler_rates_deg

    @staticmethod
    def _wrap_error(target: float, current: float) -> float:
        """Shortest-path angle error with 360° wrapping → [-180, 180]."""
        error = (target % 360) - (current % 360)
        if error > 180: error -= 360
        elif error < -180: error += 360
        return error
