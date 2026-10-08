#!/usr/bin/env python3
"""
Terrain robustness test: run baseline controller on non-flat terrain.

Tests PID stability on slopes, bumps, steps, and rough surfaces using
MuJoCo heightfields.

Usage:
    mjpython control/run_terrain_test.py --terrain slope --slope-angle 5 --headless --plot
    mjpython control/run_terrain_test.py --terrain wavy --amplitude 0.05 --drive-x 0.3 --headless --plot
    mjpython control/run_terrain_test.py --terrain rough --roughness 0.02
    mjpython control/run_terrain_test.py --terrain steps --step-height 0.03 --headless --plot
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
from control.trajectory import LineTrajectory, TrajectoryTracker
from terrain import (
    FlatTerrain, SlopeTerrain, WavyTerrain, StepTerrain, RoughTerrain,
)


def make_terrain(args):
    """Create terrain from CLI args."""
    # Don't pass elevation_max — each terrain auto-computes it
    common = dict(
        nrow=args.hfield_res, ncol=args.hfield_res,
        size_x=args.hfield_size, size_y=args.hfield_size,
    )
    if args.terrain == 'flat':
        return FlatTerrain()
    elif args.terrain == 'slope':
        return SlopeTerrain(
            angle_deg=args.slope_angle, direction=args.slope_dir, **common)
    elif args.terrain == 'wavy':
        return WavyTerrain(
            amplitude=args.amplitude, wavelength=args.wavelength, **common)
    elif args.terrain == 'steps':
        return StepTerrain(
            step_height=args.step_height,
            num_steps=args.num_steps, **common)
    elif args.terrain == 'rough':
        return RoughTerrain(
            roughness=args.roughness, seed=args.seed, **common)
    else:
        raise ValueError(f"Unknown terrain: {args.terrain}")


def run_headless(sim, controller, duration):
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id
    log = {'t': [], 'rpy': [], 'ctrl': [], 'pos': [], 'pre_alloc': []}
    ctrl = np.zeros(6)
    t_start = sim.time  # zero-reference after settle

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)

        if i % 20 == 0:
            imu = sim.get_imu()
            log['t'].append(sim.time - t_start)
            log['rpy'].append(imu['euler'].copy())
            log['ctrl'].append(ctrl.copy())
            log['pos'].append(sim.data.xpos[shell_id].copy())
            log['pre_alloc'].append(controller.last_pre_alloc.copy())

        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            pos = sim.data.xpos[shell_id]
            print(f"\rt={sim.time - t_start:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_headless_traj(sim, controller, tracker, duration, traj_start_time):
    """Headless run with trajectory tracking on terrain."""
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id
    log = {'t': [], 'rpy': [], 'ctrl': [], 'pos': [], 'pre_alloc': [],
           'ref': [], 'error': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start_time

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

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


def run_viewer(sim, controller, tracker=None, traj_start_time=0):
    """Interactive viewer. If tracker is provided, uses trajectory tracking."""
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

                if tracker is not None:
                    ball_pos = sim.data.xpos[shell_id][:2].copy()
                    traj_t = sim.time - traj_start_time
                    vx, vy, ref_x, ref_y = tracker.compute(
                        traj_t, ball_pos[0], ball_pos[1])
                    controller.set_drive(vx, vy)

                ctrl = controller.update(sim.get_imu(), ball_vel)
                sim.set_ctrl(ctrl)

            if step_i % int(0.5 / dt) == 0:
                r, p, y = sim.get_imu()['euler']
                pos = sim.data.xpos[shell_id]
                if tracker is not None:
                    traj_t = sim.time - traj_start_time
                    rx, ry = tracker.traj.position(traj_t)
                    err = np.hypot(pos[0] - rx, pos[1] - ry)
                    print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                          f"err={err:.3f}m  pos=({pos[0]:+.2f},{pos[1]:+.2f})",
                          end="", flush=True)
                else:
                    print(f"\rt={sim.time:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                          f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})",
                          end="", flush=True)

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


def plot_log(log, terrain_name):
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
    pre_alloc = log['pre_alloc']  # [roll, pitch, yaw, drive_x, drive_y]

    # Unwrap angles
    for i in range(3):
        rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))

    print(f"Saving terrain plots to {plot_dir}/")

    # ---- Fig 1: Angle Tracking Error (stability) ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'Angle Stability: {terrain_name} terrain', fontsize=13)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
        axes[i].plot(t, rpy[:, i], color=colors[i], alpha=0.8)
        mean_val = np.mean(rpy[:, i])
        std_val = np.std(rpy[:, i])
        axes[i].axhline(mean_val, color=colors[i], ls='--', alpha=0.4)
        axes[i].fill_between(t, mean_val - std_val, mean_val + std_val,
                             color=colors[i], alpha=0.1)
        axes[i].set_ylabel(f'{name} (deg)')
        axes[i].set_title(f'{name}  (mean={mean_val:+.2f}, std={std_val:.2f}, '
                          f'max={np.max(np.abs(rpy[:, i])):.2f})')
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    fig.tight_layout()
    _save_fig(fig, f'terrain_{terrain_name}_angle_stability', plot_dir)

    # ---- Fig 2: Pre-allocation Control Signals ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'Pre-allocation Control Signals: {terrain_name} terrain', fontsize=13)

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
    _save_fig(fig, f'terrain_{terrain_name}_pre_alloc_signals', plot_dir)

    _safe_show(plt)


MOTOR_NAMES = ['M0 (Yaw+)', 'M1 (Yaw-)', 'M2 (Roll+)', 'M3 (Roll-)',
               'M4 (Pitch+)', 'M5 (Pitch-)']
MOTOR_COLORS = ['tab:blue', 'tab:cyan', 'tab:red', 'tab:pink',
                'tab:green', 'tab:olive']


def plot_motors(log, terrain_name, motor_ids):
    """Plot selected motor ctrl signals in a single figure."""
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
    ctrl = log['ctrl']

    fig, ax = plt.subplots(figsize=(12, 5))
    for mid in motor_ids:
        ax.plot(t, ctrl[:, mid], label=MOTOR_NAMES[mid],
                color=MOTOR_COLORS[mid], alpha=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Command (normalized)')
    ids_str = ','.join(str(m) for m in motor_ids)
    ax.set_title(f'Motor Ctrl Signals [{ids_str}]: {terrain_name} terrain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, f'terrain_{terrain_name}_motors_{ids_str}', plot_dir)
    _safe_show(plt)


def plot_terrain_trajectory(log, terrain, terrain_name, prefix='terrain'):
    """Combined terrain heightmap + trajectory overlay plot."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
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
    pos = log['pos']
    ref = log['ref']
    error = log['error']

    # Generate terrain heightmap in meters (centered like populate())
    heights_m = terrain.generate_heights() * terrain.elevation_max
    cr, cc = terrain.nrow // 2, terrain.ncol // 2
    heights_m -= heights_m[cr, cc]
    x_coords = np.linspace(-terrain.size_x, terrain.size_x, terrain.ncol)
    y_coords = np.linspace(-terrain.size_y, terrain.size_y, terrain.nrow)

    # Bounding box of actual path + independent padding per axis
    all_x = np.concatenate([pos[:, 0], ref[:, 0]])
    all_y = np.concatenate([pos[:, 1], ref[:, 1]])
    pad_x = max(2.0, 0.10 * np.ptp(all_x))
    pad_y = max(2.0, 0.30 * max(np.ptp(all_y), 0.5))
    x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
    y_min, y_max = all_y.min() - pad_y, all_y.max() + pad_y

    # Crop terrain arrays to bounding box
    xi_min = max(0, np.searchsorted(x_coords, x_min) - 1)
    xi_max = min(len(x_coords), np.searchsorted(x_coords, x_max) + 1)
    yi_min = max(0, np.searchsorted(y_coords, y_min) - 1)
    yi_max = min(len(y_coords), np.searchsorted(y_coords, y_max) + 1)
    x_crop = x_coords[xi_min:xi_max]
    y_crop = y_coords[yi_min:yi_max]
    h_crop = heights_m[yi_min:yi_max, xi_min:xi_max]

    print(f"Saving terrain trajectory plots to {plot_dir}/")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    # ---- Top: Terrain heightmap + path overlay ----
    ax = axes[0]
    Xg, Yg = np.meshgrid(x_crop, y_crop)
    pcm = ax.pcolormesh(Xg, Yg, h_crop, cmap='gist_earth', shading='auto')
    cbar_h = fig.colorbar(pcm, ax=ax, label='Elevation (m)', shrink=0.5,
                          pad=0.01, aspect=30)

    # Contour lines for depth cue
    if np.ptp(h_crop) > 1e-6:
        n_levels = min(8, max(3, int(np.ptp(h_crop) / 0.005)))
        ax.contour(Xg, Yg, h_crop, levels=n_levels, colors='k',
                   linewidths=0.3, alpha=0.3)

    # Reference line
    ax.plot(ref[:, 0], ref[:, 1], 'w--', linewidth=2, label='Reference',
            zorder=5)

    # Actual path color-coded by error
    err_norm = Normalize(vmin=0, vmax=max(error.max(), 1e-3))
    err_cmap = plt.cm.coolwarm
    for j in range(len(pos) - 1):
        ax.plot(pos[j:j+2, 0], pos[j:j+2, 1],
                color=err_cmap(err_norm(error[j])), linewidth=1.8, zorder=6)
    sm = ScalarMappable(norm=err_norm, cmap=err_cmap)
    sm.set_array([])
    cbar_e = fig.colorbar(sm, ax=ax, label='Lateral Error (m)',
                          shrink=0.5, pad=0.04, aspect=30)

    # Start / end markers
    ax.plot(pos[0, 0], pos[0, 1], 'o', color='lime', markersize=10,
            markeredgecolor='k', markeredgewidth=1.2, label='Start', zorder=7)
    ax.plot(pos[-1, 0], pos[-1, 1], 's', color='orange', markersize=10,
            markeredgecolor='k', markeredgewidth=1.2, label='End', zorder=7)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Trajectory on {terrain_name} Terrain')
    ax.legend(loc='upper left')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ---- Bottom: Tracking error vs time ----
    ax2 = axes[1]
    ax2.plot(t, error, color='tab:blue', alpha=0.8, linewidth=1.0)
    mean_err = np.mean(error)
    std_err = np.std(error)
    max_err = np.max(error)
    ax2.axhline(mean_err, color='tab:red', ls='--', alpha=0.6, label='Mean')
    ax2.fill_between(t, max(mean_err - std_err, 0), mean_err + std_err,
                     color='tab:blue', alpha=0.1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Tracking Error (m)')
    ax2.set_title(f'Tracking Error  (mean={mean_err:.3f}m, '
                  f'max={max_err:.3f}m, std={std_err:.3f}m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, f'{prefix}_{terrain_name}_trajectory', plot_dir)
    _safe_show(plt)


def main():
    parser = argparse.ArgumentParser(
        description="Terrain robustness test with cascaded PID")
    # Terrain selection
    parser.add_argument('--terrain', type=str, default='flat',
                        choices=['flat', 'slope', 'wavy', 'steps', 'rough'],
                        help='Terrain type')
    # Terrain-specific params
    parser.add_argument('--slope-angle', type=float, default=5.0,
                        help='Slope angle in degrees')
    parser.add_argument('--slope-dir', type=str, default='x',
                        choices=['x', 'y'], help='Slope direction')
    parser.add_argument('--amplitude', type=float, default=0.05,
                        help='Wavy terrain amplitude (m)')
    parser.add_argument('--wavelength', type=float, default=2.0,
                        help='Wavy terrain wavelength (m)')
    parser.add_argument('--step-height', type=float, default=0.03,
                        help='Step terrain height (m)')
    parser.add_argument('--step-width', type=float, default=1.0,
                        help='Step terrain width (m)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='Number of steps')
    parser.add_argument('--roughness', type=float, default=0.02,
                        help='Rough terrain roughness (m)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for rough terrain')
    # Heightfield params
    parser.add_argument('--hfield-res', type=int, default=600,
                        help='Heightfield resolution (nrow=ncol)')
    parser.add_argument('--hfield-size', type=float, default=100.0,
                        help='Heightfield half-extent (m)')
    parser.add_argument('--elevation-max', type=float, default=0.5,
                        help='Max elevation scale (m)')
    # Trajectory tracking
    parser.add_argument('--trajectory', action='store_true',
                        help='Enable line trajectory tracking mode')
    parser.add_argument('--speed', type=float, default=0.5,
                        help='Travel speed along line (m/s)')
    parser.add_argument('--line-length', type=float, default=30.0,
                        help='Line trajectory length (m)')
    parser.add_argument('--line-angle', type=float, default=0.0,
                        help='Line direction in degrees from +X axis')
    # Drive params (used when --trajectory is not set)
    parser.add_argument('--drive-x', type=float, default=0.5,
                        help='Target ball velocity in X (m/s)')
    parser.add_argument('--drive-y', type=float, default=0.5,
                        help='Target ball velocity in Y (m/s)')
    # Simulation params
    parser.add_argument('--duration', type=float, default=100.0,
                        help='Simulation duration (s)')
    parser.add_argument('--settle', type=float, default=0.5,
                        help='Settling time (s)')
    parser.add_argument('--pid-rate', type=float, default=50.0,
                        help='PID update rate (Hz)')
    parser.add_argument('--mu-roll', type=float, default=0.01,
                        help='Rolling friction coefficient')
    parser.add_argument('--unidirectional', action='store_true',
                        help='Unidirectional motor mixing (no reverse)')
    parser.add_argument('--yaw-compensate', action='store_true',
                        help='Rotate drive commands from world to body frame using current yaw')
    parser.add_argument('--plot-motors', type=int, nargs='+', default=None,
                        metavar='ID',
                        help='Motor indices to plot (e.g. --plot-motors 0 1 4 5)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot or args.plot_motors:
        args.headless = True
    if args.plot_motors:
        for mid in args.plot_motors:
            if mid < 0 or mid > 5:
                parser.error(f"Motor ID must be 0-5, got {mid}")

    # Create terrain
    terrain = make_terrain(args)
    print(f"Terrain: {args.terrain}")

    # Auto-set duration for trajectory mode
    if args.trajectory:
        auto_dur = args.line_length / max(args.speed, 1e-3) + 5.0
        if args.duration == 100.0:  # default
            args.duration = auto_dur

    # Create simulation with terrain
    sim = RobotSimulation(mu_roll=args.mu_roll, terrain=terrain, force_max=5)
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

    if args.trajectory:
        # Trajectory tracking mode
        angle_rad = np.radians(args.line_angle)
        end_x = args.line_length * np.cos(angle_rad)
        end_y = args.line_length * np.sin(angle_rad)
        traj = LineTrajectory(start=(0, 0), end=(end_x, end_y),
                              speed=args.speed)
        tracker = TrajectoryTracker(traj, dt=1.0/args.pid_rate)
        traj_start_time = sim.time

        print(f"Running line trajectory (length={args.line_length}m, "
              f"angle={args.line_angle}°, speed={args.speed}m/s, "
              f"duration={args.duration:.1f}s)...")

        if args.headless:
            log = run_headless_traj(sim, controller, tracker,
                                    args.duration - args.settle,
                                    traj_start_time)
            # Tracking stats
            error = log['error']
            n = len(error) // 4
            print(f"\nTracking stats (after transient):")
            print(f"  Error: mean={np.mean(error[n:]):.3f}m  "
                  f"std={np.std(error[n:]):.3f}m  "
                  f"max={np.max(error[n:]):.3f}m")
            rpy = log['rpy']
            for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
                arr = rpy[n:, i]
                print(f"  {name:6s}: mean={np.mean(arr):+.2f}  "
                      f"std={np.std(arr):.2f}  max={np.max(np.abs(arr)):.2f}")
            if args.plot:
                plot_log(log, args.terrain)
                plot_terrain_trajectory(log, terrain, args.terrain)
            if args.plot_motors:
                plot_motors(log, args.terrain, args.plot_motors)
        else:
            run_viewer(sim, controller, tracker=tracker,
                       traj_start_time=traj_start_time)
    else:
        # Original constant-velocity mode
        controller.set_drive(vx=args.drive_x, vy=args.drive_y)

        if args.headless:
            log = run_headless(sim, controller, args.duration - args.settle)
            # Summary stats
            rpy = log['rpy']
            n = len(rpy) // 2
            print(f"\nStability stats (second half):")
            for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
                arr = rpy[n:, i]
                print(f"  {name:6s}: mean={np.mean(arr):+.2f}  "
                      f"std={np.std(arr):.2f}  max={np.max(np.abs(arr)):.2f}")
            pos = log['pos']
            print(f"  Drift:  dX={pos[-1,0]-pos[0,0]:+.3f}  "
                  f"dY={pos[-1,1]-pos[0,1]:+.3f}  "
                  f"dZ={pos[-1,2]-pos[0,2]:+.3f} m")
            if args.plot:
                plot_log(log, args.terrain)
            if args.plot_motors:
                plot_motors(log, args.terrain, args.plot_motors)
        else:
            run_viewer(sim, controller)


if __name__ == "__main__":
    main()
