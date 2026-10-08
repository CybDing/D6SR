"""
Lightweight reusable PID controller.

Use as PID (ki>0) for velocity→force, or PD (ki=0) for position→velocity.
"""

import numpy as np


class SimplePID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.02,
                 output_limit=1.0, integral_limit=0.5, d_filter_alpha=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.d_alpha = d_filter_alpha
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filtered = 0.0
        self._prev_output = 0.0

    def step(self, error: float) -> float:
        # P
        p = self.kp * error
        # I with conditional anti-windup
        saturated = abs(self._prev_output) >= self.output_limit * 0.99
        if not (saturated and error * self.integral > 0):
            self.integral = float(np.clip(
                self.integral + error * self.dt,
                -self.integral_limit, self.integral_limit))
        i = self.ki * self.integral
        # D with EMA filter
        d_raw = (error - self.prev_error) / max(self.dt, 1e-6)
        self.d_filtered = self.d_alpha * d_raw + (1 - self.d_alpha) * self.d_filtered
        d = self.kd * self.d_filtered

        output = float(np.clip(p + i + d, -self.output_limit, self.output_limit))
        self.prev_error = error
        self._prev_output = output
        return output
