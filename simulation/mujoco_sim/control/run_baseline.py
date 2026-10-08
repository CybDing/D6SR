#!/usr/bin/env python3
"""
Baseline control: cascaded PID stabilization + velocity PID drive.

V1 (gimbal): attitude PIDs stabilize inner gimbal, drive via legacy mixing.
V0 (single-body): motors fixed on shell → world force rotated to body frame
    via full quaternion rotation matrix. No attitude PIDs needed (ball must
    roll to move). Motor allocation shifts continuously as shell rotates.

Usage:
    # V1 (default)
    mjpython control/run_baseline.py --drive-x 0.5

    # V0 (auto-detected from model)
    mjpython control/run_baseline.py --mjcf models/robot_v0_mujoco.xml --drive-x 0.3

    # Headless with plot
    python control/run_baseline.py --mjcf models/robot_v0_mujoco.xml \
        --headless --plot --drive-x 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation import RobotSimulation
from control.cascaded_pid import CascadedPID, CascadedPIDConfig
from control.simple_pid import SimplePID
from control.allocation import allocate_wrench, quat_to_rotmat


# ======================================================================
# PID configs (V1 attitude control)
# ======================================================================

FORCE_SCALING_FACTOR = 0.4
DRIVE_KP = 1.0
DRIVE_KI = 0.5
DRIVE_KD = 0.3
DRIVE_OUTPUT_LIMIT = 0.8
UNIDIRECTIONAL = 2
GRAVITATIONAL_OFFSET = 0

def make_roll_config(dt=0.02):
    return CascadedPIDConfig(
        outer_kp=0.8 * FORCE_SCALING_FACTOR, outer_ki=0.05 * FORCE_SCALING_FACTOR, outer_kd=0.00 * FORCE_SCALING_FACTOR,
        outer_max_output=60.0, outer_deadband=0.0,
        inner_kp=0.0008 * 5 , inner_ki=0.0000, inner_kd=0.00005 * 5,
        inner_max_output=0.8,
        dt=dt, velocity_slew_rate=30.0, output_slew_rate=0.01,
        gravity_comp_gain=0.1365 * 5 * FORCE_SCALING_FACTOR * UNIDIRECTIONAL * GRAVITATIONAL_OFFSET,
        grav_sign=1,
    )

def make_pitch_config(dt=0.02):
    return CascadedPIDConfig(
        outer_kp=0.8 * FORCE_SCALING_FACTOR, outer_ki=0.05 * FORCE_SCALING_FACTOR, outer_kd=0.0 * FORCE_SCALING_FACTOR,
        outer_max_output=60.0, outer_deadband=0.0,
        inner_kp=0.0008 * 5, inner_ki=0.0000, inner_kd=0.00005 * 5,
        inner_max_output=0.8,
        dt=dt, velocity_slew_rate=30.0, output_slew_rate=0.01,
        gravity_comp_gain=0.1365 * 5 * FORCE_SCALING_FACTOR * UNIDIRECTIONAL * GRAVITATIONAL_OFFSET,
        grav_sign=1, outer_integral_limit=100,
    )

def make_yaw_config(dt=0.02):
    return CascadedPIDConfig(
        outer_kp=0.8 * FORCE_SCALING_FACTOR, outer_ki=0.05 * FORCE_SCALING_FACTOR, outer_kd=0.01 * FORCE_SCALING_FACTOR,
        outer_max_output=30.0, outer_deadband=0.0,
        inner_kp=0.0008 * 5 * FORCE_SCALING_FACTOR, inner_ki=0.000 * FORCE_SCALING_FACTOR, inner_kd=0.0001 * 5 * FORCE_SCALING_FACTOR,
        inner_max_output=0.8,
        dt=dt, velocity_slew_rate=50.0, output_slew_rate=0.01,
    )


# ======================================================================
# Motor mixing (V1 legacy)
# ======================================================================

def mix_motors(ro, po, yo, xd, yd, unidirectional=False):
    """Mix attitude torques + drive signals to 6 motor commands (V1 only)."""
    if not unidirectional:
        return np.array([
            +yo + yd,   # M0
            -yo + yd,   # M1
            +ro,        # M2
            -ro,        # M3
            +po + xd,   # M4
            -po + xd,   # M5
        ])

    if yd >= 0:
        m0 = max(yo, 0) + yd
        m1 = max(-yo, 0) + yd
    else:
        m0 = min(yo, 0) + yd
        m1 = min(-yo, 0) + yd
    m2 = max(ro, 0)
    m3 = max(-ro, 0)
    if xd >= 0:
        m4 = max(po, 0) + xd
        m5 = max(-po, 0) + xd
    else:
        m4 = min(po, 0) + xd
        m5 = min(-po, 0) + xd
    return np.array([m0, m1, m2, m3, m4, m5])


# ======================================================================
# Controller
# ======================================================================

class BaselineController:
    """Unified controller for V0 (single-body) and V1 (gimbal) robots.

    V0 mode (auto-detected):
        - Velocity PID → world-frame force
        - R_bw^T rotates to body frame (full quaternion)
        - T^{-1} allocates to 6 motors
        - No attitude PIDs (ball must roll to move)

    V1 mode:
        - Cascaded attitude PIDs (roll/pitch/yaw)
        - Velocity PID drive
        - Legacy motor mixing (body ≈ world frame)
    """

    def __init__(self, pid_rate=50.0, v0_mode=False,
                 motor_radius=0.13, force_scale=2.0, torque_scale=None,
                 # V1-only options
                 unidirectional=False, yaw_compensate=False,
                 allocation_mode='legacy', use_imu_rotation=None,
                 drive_only=False, drive_force_mode=False,
                 drive_output_limit=None, euler_rate_transform=False,
                 attitude_gain_scale=1.0, attitude_gains=None):
        self.pid_rate = pid_rate
        self.pid_dt = 1.0 / pid_rate
        self.v0_mode = v0_mode
        self.motor_radius = motor_radius
        self.force_scale = force_scale
        self.torque_scale = torque_scale

        # V1-specific
        self.unidirectional = unidirectional
        self.yaw_compensate = yaw_compensate
        self.allocation_mode = allocation_mode
        self.use_imu_rotation = use_imu_rotation
        if self.allocation_mode == 'matrix' and self.use_imu_rotation is None:
            self.use_imu_rotation = True
        if self.use_imu_rotation is None:
            self.use_imu_rotation = False
        self.drive_only = drive_only
        self.drive_force_mode = drive_force_mode
        self.euler_rate_transform = euler_rate_transform

        # Attitude PIDs (V1 only, ignored in V0)
        rcfg = make_roll_config(self.pid_dt)
        pcfg = make_pitch_config(self.pid_dt)
        ycfg = make_yaw_config(self.pid_dt)
        # (legacy) uniform scale of all gains — a crude proxy, kept for
        # back-compat but NOT a real re-tune.
        if attitude_gain_scale != 1.0:
            for cfg in (rcfg, pcfg, ycfg):
                for f in ('outer_kp', 'outer_ki', 'outer_kd',
                          'inner_kp', 'inner_ki', 'inner_kd'):
                    setattr(cfg, f, getattr(cfg, f) * attitude_gain_scale)
        # Proper re-tune: an explicit per-axis PID parameter set. This is the
        # base controller "initialized with a given parameter dict". Any field
        # present overrides the committed default; missing fields keep it.
        #   attitude_gains = {'roll': {'inner_kp':..., 'outer_kp':..., ...},
        #                     'pitch': {...}, 'yaw': {...}}
        if attitude_gains:
            for axis, cfg in (('roll', rcfg), ('pitch', pcfg), ('yaw', ycfg)):
                for field, val in (attitude_gains.get(axis) or {}).items():
                    if not hasattr(cfg, field):
                        raise KeyError(f"unknown PID field '{field}' for {axis}")
                    setattr(cfg, field, val)
        self.attitude_gain_scale = attitude_gain_scale
        self.attitude_gains = attitude_gains
        self.roll_pid = CascadedPID(rcfg, 'Roll')
        self.pitch_pid = CascadedPID(pcfg, 'Pitch')
        self.yaw_pid = CascadedPID(ycfg, 'Yaw')

        # Velocity drive PIDs (both V0 and V1)
        vel_limit = DRIVE_OUTPUT_LIMIT if drive_output_limit is None else drive_output_limit
        self.vel_pid_x = SimplePID(DRIVE_KP, DRIVE_KI, DRIVE_KD,
                                   dt=self.pid_dt, output_limit=vel_limit)
        self.vel_pid_y = SimplePID(DRIVE_KP, DRIVE_KI, DRIVE_KD,
                                   dt=self.pid_dt, output_limit=vel_limit)
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.roll_enabled = False
        self.pitch_enabled = False
        self.yaw_enabled = False
        self.last_pre_alloc = np.zeros(5)

    @property
    def enabled(self):
        return self.roll_enabled and self.pitch_enabled and self.yaw_enabled

    @enabled.setter
    def enabled(self, val):
        self.roll_enabled = self.pitch_enabled = self.yaw_enabled = val

    def set_target(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll_pid.set_target(roll)
        self.pitch_pid.set_target(pitch)
        self.yaw_pid.set_target(yaw)

    def set_drive(self, vx=0.0, vy=0.0):
        """Set target ball velocity (m/s) in world frame."""
        self.target_vx = vx
        self.target_vy = vy

    def reset(self):
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.vel_pid_x.reset()
        self.vel_pid_y.reset()

    def update(self, imu: dict, ball_vel=(0.0, 0.0)) -> np.ndarray:
        # Velocity PID → world-frame drive command
        xd = self.vel_pid_x.step(self.target_vx - ball_vel[0])
        yd = self.vel_pid_y.step(self.target_vy - ball_vel[1])

        if self.v0_mode:
            return self._update_v0(imu, xd, yd)
        else:
            return self._update_v1(imu, xd, yd)

    def _update_v0(self, imu: dict, xd: float, yd: float) -> np.ndarray:
        """V0: rotate world-frame force to body frame, allocate to motors.

        The key insight: motors are fixed on the shell. As the shell rolls,
        motor thrust directions rotate. We must transform the desired
        world-frame force into the current body frame so the allocation
        matrix (which maps body-frame wrench → motors) works correctly.

        This means motor commands continuously shift between pairs as the
        shell orientation changes — that IS the rotation compensation.
        """
        # World-frame desired wrench
        force_world = np.array([xd, yd, 0.0])
        torque_world = np.zeros(3)  # no attitude torques for V0

        # Rotate world → body using full quaternion
        R_bw = quat_to_rotmat(imu['quat'])  # body-to-world rotation
        force_body = R_bw.T @ force_world     # world-to-body: R_wb = R_bw^T
        torque_body = R_bw.T @ torque_world

        self.last_pre_alloc = np.array([0.0, 0.0, 0.0, xd, yd])

        ctrl = allocate_wrench(
            force_body, torque_body,
            motor_radius=self.motor_radius,
            force_scale=self.force_scale,
            torque_scale=self.torque_scale,
        )
        return np.clip(ctrl, -1.0, 1.0)

    def _update_v1(self, imu: dict, xd: float, yd: float) -> np.ndarray:
        """V1: cascaded attitude PIDs + legacy motor mixing."""
        r, p, y = imu['euler']
        gx, gy, gz = np.degrees(imu['gyro'])

        if self.drive_only:
            ro = po = yo = 0.0
        elif self.euler_rate_transform:
            # Split outer/inner loops with Euler-rate → body-rate transform.
            # Outer loops produce Euler angle rate targets (ṙ, ṗ, ẏ).
            er_r = self.roll_pid.outer_step(r)  if self.roll_enabled  else 0.0
            er_p = self.pitch_pid.outer_step(p) if self.pitch_enabled else 0.0
            er_y = self.yaw_pid.outer_step(y)   if self.yaw_enabled   else 0.0

            # E(φ,θ) converts Euler rates to body angular velocity targets.
            euler_rates = np.array([er_r, er_p, er_y])
            body_rates = CascadedPID.euler_rate_to_body_rate(euler_rates, r, p)

            # Inner loops track body angular velocity.
            ro = self.roll_pid.inner_step(body_rates[0],  gx, r) if self.roll_enabled  else 0.0
            po = self.pitch_pid.inner_step(body_rates[1], gy, p) if self.pitch_enabled else 0.0
            yo = self.yaw_pid.inner_step(body_rates[2],   gz, y) if self.yaw_enabled   else 0.0
        else:
            ro = self.roll_pid.step(r, gx) if self.roll_enabled else 0.0
            po = self.pitch_pid.step(p, gy) if self.pitch_enabled else 0.0
            yo = self.yaw_pid.step(y, gz) if self.yaw_enabled else 0.0

        if self.allocation_mode == 'legacy' and self.yaw_compensate:
            # yaw_comp_freeze: None -> continuous compensation (legacy);
            # a number (deg) -> piecewise-frozen compensation, updated only
            # at explicit re-calibration events (post-flip). Models a one-off
            # heading re-calibration instead of a continuous overlay.
            y_used = self.yaw_comp_freeze \
                if getattr(self, 'yaw_comp_freeze', None) is not None else y
            yaw_rad = np.radians(y_used)
            cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
            xd, yd = xd * cos_y + yd * sin_y, -xd * sin_y + yd * cos_y

        if self.drive_force_mode and self.allocation_mode == 'legacy':
            scale = max(self.force_scale, 1e-9)
            xd /= scale
            yd /= scale

        self.last_pre_alloc = np.array([ro, po, yo, xd, yd])

        if self.allocation_mode == 'matrix':
            force_world = np.array([xd, yd, 0.0], dtype=float)
            torque_world = np.array([ro, po, yo], dtype=float)
            if self.use_imu_rotation and 'quat' in imu:
                if self.use_imu_rotation == 'yaw':
                    yaw_rad = np.radians(y)
                    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
                    R_yaw = np.array([
                        [cos_y, -sin_y, 0.0],
                        [sin_y,  cos_y, 0.0],
                        [0.0,    0.0,   1.0],
                    ])
                    force_body = R_yaw.T @ force_world
                    torque_body = R_yaw.T @ torque_world
                else:
                    R_bw = quat_to_rotmat(imu['quat'])
                    force_body = R_bw.T @ force_world
                    torque_body = R_bw.T @ torque_world
            else:
                force_body = force_world
                torque_body = torque_world
            ctrl = allocate_wrench(
                force_body, torque_body,
                motor_radius=self.motor_radius,
                force_scale=self.force_scale,
                torque_scale=self.torque_scale,
            )
        elif self.allocation_mode == 'legacy':
            ctrl = mix_motors(ro, po, yo, xd, yd, self.unidirectional)
        else:
            raise ValueError(f"Unknown allocation_mode: {self.allocation_mode!r}")
        return np.clip(ctrl, -1.0, 1.0)


# ======================================================================
# Run modes
# ======================================================================

def get_ball_vel(model, data, body_id):
    """Get ball translational velocity (vx, vy) in world frame."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    return vel[3], vel[4]


def run_headless(sim, controller, duration):
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id
    log = {'t': [], 'rpy': [], 'ctrl': [], 'pos': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)
        if i % 20 == 0:
            imu = sim.get_imu()
            log['t'].append(sim.time)
            log['rpy'].append(imu['euler'].copy())
            log['ctrl'].append(ctrl.copy())
            log['pos'].append(sim.data.xpos[shell_id].copy())
        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            pos = sim.data.xpos[shell_id]
            ctrl_str = ','.join(f'{c:+.2f}' for c in ctrl)
            print(f"\rt={sim.time:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"pos=({pos[0]:+.3f},{pos[1]:+.3f})  "
                  f"ctrl=[{ctrl_str}]", end="", flush=True)
        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_viewer(sim, controller):
    import mujoco.viewer
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    shell_id = sim._shell_body_id
    ctrl = np.zeros(6)
    step_i = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            t0 = time.perf_counter()
            if step_i % steps_per_pid == 0:
                ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
                ctrl = controller.update(sim.get_imu(), ball_vel)
                sim.set_ctrl(ctrl)
            if step_i % int(0.5 / dt) == 0:
                r, p, y = sim.get_imu()['euler']
                ctrl_str = ','.join(f'{c:+.2f}' for c in ctrl)
                print(f"\rt={sim.time:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                      f"ctrl=[{ctrl_str}]",
                      end="", flush=True)
            sim.step()
            viewer.sync()
            step_i += 1
            sl = dt - (time.perf_counter() - t0)
            if sl > 0: time.sleep(sl)
    print()


def _safe_show(plt):
    backend = plt.get_backend().lower()
    if 'agg' not in backend:
        try:
            plt.show()
        except Exception:
            pass


def plot_log(log):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    t, rpy, ctrl, pos = log['t'], log['rpy'], log['ctrl'], log['pos']
    for i in range(3):
        rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i, name in enumerate(['Roll','Pitch','Yaw']):
        axes[0].plot(t, rpy[:, i], label=name)
    axes[0].set_ylabel('Angle (deg)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Shell Orientation')

    motor_labels = ['M0 yaw', 'M1 yaw', 'M2 roll', 'M3 roll', 'M4 pitch', 'M5 pitch']
    colors = ['#3344cc', '#5566ee', '#cc3333', '#ee5555', '#33aa33', '#55cc55']
    for i in range(6):
        axes[1].plot(t, ctrl[:, i], label=motor_labels[i], color=colors[i], alpha=0.8)
    axes[1].set_ylabel('Motor cmd')
    axes[1].legend(ncol=3, fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Motor Allocation (motors shift as shell rotates)')

    pair_yaw = np.abs(ctrl[:, 0]) + np.abs(ctrl[:, 1])
    pair_roll = np.abs(ctrl[:, 2]) + np.abs(ctrl[:, 3])
    pair_pitch = np.abs(ctrl[:, 4]) + np.abs(ctrl[:, 5])
    axes[2].plot(t, pair_yaw, label='Yaw pair (M0+M1)', color='#3344cc')
    axes[2].plot(t, pair_roll, label='Roll pair (M2+M3)', color='#cc3333')
    axes[2].plot(t, pair_pitch, label='Pitch pair (M4+M5)', color='#33aa33')
    axes[2].set_ylabel('Pair |activity|')
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Motor Pair Activity (pairs cycle as shell rotates)')

    axes[3].plot(t, pos[:, 0], label='X'); axes[3].plot(t, pos[:, 1], label='Y')
    axes[3].set_ylabel('Pos (m)'); axes[3].set_xlabel('Time (s)')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / 'baseline_result.png'
    plt.savefig(str(out), dpi=150); print(f"Saved {out}")
    _safe_show(plt)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline: cascaded PID + drive")
    parser.add_argument('--drive-x', type=float, default=0.7,
                        help='target ball velocity in X (m/s)')
    parser.add_argument('--drive-y', type=float, default=0,
                        help='target ball velocity in Y (m/s)')
    parser.add_argument('--duration', type=float, default=100.0)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--pid-rate', type=float, default=50.0)
    parser.add_argument('--mu-roll', type=float, default=0.01)
    parser.add_argument('--mjcf', type=str, default=None,
                        help='Path to MJCF (default: robot_v1_mujoco.xml)')
    parser.add_argument('--shell-body', type=str, default=None)
    parser.add_argument('--base-body', type=str, default='base_link')
    parser.add_argument('--force-max', type=float, default=5)
    parser.add_argument('--force-scale', type=float, default=2.0)
    parser.add_argument('--torque-scale', type=float, default=None)
    # V1-specific (ignored for V0)
    parser.add_argument('--alloc', choices=['legacy', 'matrix'], default='legacy')
    parser.add_argument('--drive-only', action='store_true')
    parser.add_argument('--drive-force', action='store_true')
    parser.add_argument('--drive-output-limit', type=float, default=None)
    parser.add_argument('--imu-alloc', dest='imu_alloc', action='store_true', default=None)
    parser.add_argument('--no-imu-alloc', dest='imu_alloc', action='store_false')
    parser.add_argument('--imu-alloc-yaw', action='store_true')
    parser.add_argument('--unidirectional', action='store_true')
    parser.add_argument('--yaw-compensate', action='store_true')
    parser.add_argument('--euler-rate-transform', action='store_true',
                        help='Apply E(roll,pitch) Euler-rate→body-rate transform in inner loop')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run without viewer (default: True)')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='Run with interactive viewer (requires mjpython on macOS)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot: args.headless = True

    sim = RobotSimulation(
        mjcf_path=args.mjcf,
        mu_roll=args.mu_roll,
        force_max=args.force_max,
        shell_body_name=args.shell_body,
        base_body_name=args.base_body,
    )
    sim.reset()

    # Auto-detect V0: single-body model (shell == base, no gimbal)
    is_v0 = (sim._shell_body_name == sim._base_body_name)
    if is_v0:
        print("\n=== V0 mode: motors fixed on shell ===")
        print("  World force → R_bw^T → body force → T^{-1} → motors")
        print("  Motor commands shift between pairs as shell rotates\n")

    controller = BaselineController(
        pid_rate=args.pid_rate,
        v0_mode=is_v0,
        motor_radius=sim._motor_radius,
        force_scale=args.force_scale,
        torque_scale=args.torque_scale,
        # V1 options (ignored when v0_mode=True)
        unidirectional=args.unidirectional,
        yaw_compensate=args.yaw_compensate,
        allocation_mode=args.alloc,
        use_imu_rotation='yaw' if args.imu_alloc_yaw else args.imu_alloc,
        drive_only=args.drive_only,
        drive_force_mode=args.drive_force,
        drive_output_limit=args.drive_output_limit,
        euler_rate_transform=args.euler_rate_transform,
    )

    # Settle
    for _ in range(int(args.settle / sim.dt)):
        sim.step()
    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.set_drive(vx=args.drive_x, vy=args.drive_y)
    controller.enabled = True

    if args.headless:
        log = run_headless(sim, controller, args.duration - args.settle)
        rpy = log['rpy']
        n = len(rpy) // 2
        for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
            arr = rpy[n:, i]
            print(f"  {name:6s}: mean={np.mean(arr):+.2f}  std={np.std(arr):.2f}  "
                  f"max={np.max(np.abs(arr)):.2f}")
        print(f"  Travel: dX={log['pos'][-1,0]-log['pos'][0,0]:+.3f}  "
              f"dY={log['pos'][-1,1]-log['pos'][0,1]:+.3f} m")
        if args.plot: plot_log(log)
    else:
        run_viewer(sim, controller)


if __name__ == "__main__":
    main()
