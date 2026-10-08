#!/usr/bin/env python3
"""
Trajectory tracking: position-level servo on top of velocity-controlled drive.

Control hierarchy:
    TrajectoryGenerator → (x_ref, y_ref) + (vx_ff, vy_ff)
        ↓
    TrajectoryTracker → vx_cmd = vx_ff + PD(x_ref - x_actual)
        ↓
    BaselineController.set_drive(vx_cmd, vy_cmd)
        ↓
    Velocity PID → drive motor output
        ↓
    CascadedPID → attitude stabilization → motor mixing → ctrl[0..5]

Usage:
    mjpython control/run_trajectory.py --trajectory circle --radius 2 --speed 0.3
    mjpython control/run_trajectory.py --trajectory eight --radius 1.5 --speed 0.2
    mjpython control/run_trajectory.py --trajectory line --speed 0.3 --duration 20 --headless --plot
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from control.trajectory import (
    CircleTrajectory, LineTrajectory, FigureEightTrajectory, TrajectoryTracker,
)


def make_trajectory(args):
    """Create trajectory from CLI args."""
    if args.trajectory == 'circle':
        return CircleTrajectory(
            radius=args.radius, speed=args.speed, center=(0.0, 0.0))
    elif args.trajectory == 'line':
        return LineTrajectory(
            start=(0.0, 0.0), end=(args.line_end_x, args.line_end_y),
            speed=args.speed)
    elif args.trajectory == 'eight':
        return FigureEightTrajectory(
            radius=args.radius, speed=args.speed, center=(0.0, 0.0))
    else:
        raise ValueError(f"Unknown trajectory: {args.trajectory}")


def run_headless(sim, controller, tracker, duration, traj_start_time):
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id
    log = {'t': [], 'rpy': [], 'ctrl': [], 'pos': [], 'ref': [], 'error': [],
           'pre_alloc': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            # Get ball state
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start_time

            # Trajectory tracker → velocity commands
            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            # PID update
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)

        if i % 20 == 0:
            imu = sim.get_imu()
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - traj_start_time
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['rpy'].append(imu['euler'].copy())
            log['ctrl'].append(ctrl.copy())
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)
            log['pre_alloc'].append(controller.last_pre_alloc.copy())

        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            ball_pos = sim.data.xpos[shell_id][:2]
            traj_t = sim.time - traj_start_time
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_viewer(sim, controller, tracker, traj_start_time):
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
                ball_pos = sim.data.xpos[shell_id][:2].copy()
                ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
                traj_t = sim.time - traj_start_time

                vx, vy, ref_x, ref_y = tracker.compute(
                    traj_t, ball_pos[0], ball_pos[1])
                controller.set_drive(vx, vy)
                ctrl = controller.update(sim.get_imu(), ball_vel)
                sim.set_ctrl(ctrl)

            if step_i % int(0.5 / dt) == 0:
                r, p, y = sim.get_imu()['euler']
                ball_pos = sim.data.xpos[shell_id][:2]
                traj_t = sim.time - traj_start_time
                ref_x, ref_y = tracker.traj.position(traj_t)
                err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
                print(f"\rt={sim.time:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                      f"err={err:.3f}m", end="", flush=True)

            sim.step()
            viewer.sync()
            step_i += 1
            sl = dt - (time.perf_counter() - t0)
            if sl > 0:
                time.sleep(sl)
    print()


def _safe_show(plt):
    backend = plt.get_backend().lower()
    if 'agg' not in backend:
        try:
            plt.show()
        except Exception:
            pass


def _save_fig(fig, name, plot_dir):
    out = plot_dir / f'{name}.png'
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"  Saved {out}")


def plot_log(log, trajectory_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)

    t = log['t']
    rpy = log['rpy'].copy()
    pos = log['pos']
    ref = log['ref']
    error = log['error']
    pre_alloc = log['pre_alloc']  # [roll, pitch, yaw, drive_x, drive_y]

    # Unwrap angles
    for i in range(3):
        rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))

    print(f"Saving trajectory plots to {plot_dir}/")

    # ---- Fig 1: XY Trajectory ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref[:, 0], ref[:, 1], color='tab:gray', ls='--', alpha=0.7,
            label='Reference', linewidth=1.2)
    ax.plot(pos[:, 0], pos[:, 1], color='tab:blue', alpha=0.85,
            label='Actual', linewidth=1.4)
    ax.plot(pos[0, 0], pos[0, 1], 'o', color='tab:green', markersize=8, label='Start')
    ax.plot(pos[-1, 0], pos[-1, 1], 's', color='tab:orange', markersize=8, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Trajectory: {trajectory_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    _save_fig(fig, f'traj_{trajectory_name}_xy_path', plot_dir)

    # ---- Fig 2: Tracking Error ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, error, color='tab:blue', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'Tracking Error (mean={np.mean(error):.3f}m, max={np.max(error):.3f}m)')
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f'traj_{trajectory_name}_tracking_error', plot_dir)

    # ---- Fig 3: RPY Angles ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
        ax.plot(t, rpy[:, i], label=name, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('RPY Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f'traj_{trajectory_name}_rpy_angles', plot_dir)

    # ---- Fig 4: Pre-allocation Control Signals ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'Pre-allocation Control Signals: {trajectory_name}', fontsize=13)

    # Attitude torques: roll, pitch, yaw
    for i, name in enumerate(['Roll torque', 'Pitch torque', 'Yaw torque']):
        axes[0].plot(t, pre_alloc[:, i], label=name, alpha=0.8)
    axes[0].set_ylabel('Torque (normalized)')
    axes[0].set_title('Attitude Control Torques')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drive signals: drive_x, drive_y
    axes[1].plot(t, pre_alloc[:, 3], label='Drive X', alpha=0.8)
    axes[1].plot(t, pre_alloc[:, 4], label='Drive Y', alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Drive (normalized)')
    axes[1].set_title('Drive Acceleration Signals')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, f'traj_{trajectory_name}_pre_alloc_signals', plot_dir)

    _safe_show(plt)


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory tracking with cascaded PID")
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'line', 'eight'],
                        help='Trajectory type')
    parser.add_argument('--radius', type=float, default=2.0,
                        help='Radius for circle/eight trajectories (m)')
    parser.add_argument('--speed', type=float, default=0.3,
                        help='Travel speed (m/s)')
    parser.add_argument('--line-end-x', type=float, default=5.0,
                        help='Line trajectory end X (m)')
    parser.add_argument('--line-end-y', type=float, default=0.0,
                        help='Line trajectory end Y (m)')
    parser.add_argument('--max-speed', type=float, default=1.0,
                        help='Max velocity command (m/s)')
    parser.add_argument('--duration', type=float, default=50.0,
                        help='Total simulation duration (s)')
    parser.add_argument('--settle', type=float, default=2.0,
                        help='Initial settling time (s)')
    parser.add_argument('--pid-rate', type=float, default=50.0,
                        help='PID update rate (Hz)')
    parser.add_argument('--mu-roll', type=float, default=0.01,
                        help='Rolling friction coefficient')
    parser.add_argument('--unidirectional', action='store_true',
                        help='Unidirectional motor mixing (no reverse)')
    parser.add_argument('--yaw-compensate', action='store_true',
                        help='Rotate drive commands from world to body frame using current yaw')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot:
        args.headless = True

    # Create trajectory and tracker
    traj = make_trajectory(args)
    tracker = TrajectoryTracker(traj, dt=1.0/args.pid_rate,
                                max_speed=args.max_speed)

    # Create simulation
    sim = RobotSimulation(mu_roll=args.mu_roll, force_max=5)
    sim.reset()
    controller = BaselineController(pid_rate=args.pid_rate,
                                    unidirectional=args.unidirectional,
                                    yaw_compensate=args.yaw_compensate)

    # Settle
    print(f"Settling for {args.settle}s...")
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = True

    # controller.yaw_enabled = True
    # controller.pitch_enabled = True
    # controller.roll_enabled = True

    traj_start_time = sim.time

    print(f"Running {args.trajectory} trajectory "
          f"(speed={args.speed}, duration={args.duration}s)...")

    if args.headless:
        log = run_headless(sim, controller, tracker,
                           args.duration - args.settle, traj_start_time)
        # Summary stats
        error = log['error']
        n = len(error) // 4  # skip initial transient
        print(f"\nTracking stats (after transient):")
        print(f"  Error: mean={np.mean(error[n:]):.3f}m  "
              f"std={np.std(error[n:]):.3f}m  "
              f"max={np.max(error[n:]):.3f}m")
        rpy = log['rpy']
        for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
            arr = rpy[n:, i]
            print(f"  {name:6s}: mean={np.mean(arr):+.2f}  std={np.std(arr):.2f}  "
                  f"max={np.max(np.abs(arr)):.2f}")
        if args.plot:
            plot_log(log, args.trajectory)
    else:
        run_viewer(sim, controller, tracker, traj_start_time)


if __name__ == "__main__":
    main()
