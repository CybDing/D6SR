#!/usr/bin/env python3
"""
Evaluate trained RL policy on trajectory tracking tasks.

Runs the same experiment as control/run_trajectory.py but with the RL residual
policy on top of PID. Generates separate plots for:
  1. XY trajectory path
  2. Tracking error
  3. RPY angles
  4. Pre-allocation control signals (3 attitude torques + 2 drive)

Usage:
    python rl/eval_trajectory.py --trajectory circle --radius 2 --speed 0.3
    python rl/eval_trajectory.py --trajectory eight --policy rl/checkpoints/best.pt
    python rl/eval_trajectory.py --trajectory line --speed 0.3 --duration 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.env import SphericalRobotEnv, OBS_DIM_FULL, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from control.trajectory import (
    CircleTrajectory, LineTrajectory, FigureEightTrajectory, TrajectoryTracker,
)


def make_trajectory(args):
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


def run_eval(env, agent, tracker, duration, traj_start_time):
    """Run trajectory tracking with RL policy, return log dict."""
    dt = env.sim.dt
    steps_per_pid = max(1, int(round(1.0 / env.pid_rate / dt)))
    total = int(duration / dt)
    shell_id = env.shell_id
    controller = env.controller

    log = {'t': [], 'rpy': [], 'pos': [], 'ref': [], 'error': [],
           'pre_alloc': [], 'residual': [], 'joint_angles': []}
    residual = np.zeros(env.act_dim)

    pid_step_count = 0

    for i in range(total):
        if i % steps_per_pid == 0:
            # Trajectory tracker -> velocity commands
            ball_pos = env.sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(env.sim.model, env.sim.data, shell_id)
            traj_t = env.sim.time - traj_start_time

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            # Get observation for RL
            obs = env._get_obs()

            # RL policy action
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

            # PID update
            imu = env.sim.get_imu()
            pid_ctrl = controller.update(imu, ball_vel)

            # Apply residual
            residual = env.alpha * action

            if env.unidirectional and env.act_dim == 3:
                pre = controller.last_pre_alloc
                combined_ro = pre[0] + residual[0]
                combined_po = pre[1] + residual[1]
                combined_yo = pre[2] + residual[2]
                ctrl_total = np.clip(
                    mix_motors(combined_ro, combined_po, combined_yo,
                               pre[3], pre[4], unidirectional=True),
                    -1.0, 1.0)
            else:
                residual_6 = env._mix_residual(residual)
                ctrl_total = np.clip(pid_ctrl + residual_6, -1.0, 1.0)

            env.sim.set_ctrl(ctrl_total)

            pid_step_count += 1

        if i % 20 == 0:
            imu = env.sim.get_imu()
            ball_pos = env.sim.data.xpos[shell_id][:2].copy()
            traj_t = env.sim.time - traj_start_time
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)

            log['t'].append(env.sim.time - traj_start_time)
            log['rpy'].append(imu['euler'].copy())
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)
            log['pre_alloc'].append(controller.last_pre_alloc.copy())
            log['residual'].append(residual.copy())
            j1, _ = env.sim.get_joint_state('joint1')
            j2, _ = env.sim.get_joint_state('joint2')
            j3, _ = env.sim.get_joint_state('joint3')
            log['joint_angles'].append(np.array([j1, j2, j3]))

        if i % int(0.5 / dt) == 0:
            r, p, y = env.sim.get_imu()['euler']
            ball_pos = env.sim.data.xpos[shell_id][:2]
            traj_t = env.sim.time - traj_start_time
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        env.sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


# ======================================================================
# Plotting
# ======================================================================

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
    pre_alloc = log['pre_alloc']

    for i in range(3):
        rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))

    print(f"Saving RL trajectory plots to {plot_dir}/")

    # ---- Fig 1: XY Trajectory ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref[:, 0], ref[:, 1], color='tab:gray', ls='--', alpha=0.7,
            label='Reference', linewidth=2)
    ax.plot(pos[:, 0], pos[:, 1], color='tab:blue', alpha=0.85,
            label='Actual (RL)', linewidth=1.5)
    ax.plot(pos[0, 0], pos[0, 1], 'o', color='tab:green', markersize=8, label='Start')
    ax.plot(pos[-1, 0], pos[-1, 1], 's', color='tab:orange', markersize=8, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'RL Trajectory: {trajectory_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    _save_fig(fig, f'rl_traj_{trajectory_name}_xy_path', plot_dir)

    # ---- Fig 2: Tracking Error ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, error, color='tab:blue', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'RL Tracking Error (mean={np.mean(error):.3f}m, max={np.max(error):.3f}m)')
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f'rl_traj_{trajectory_name}_tracking_error', plot_dir)

    # ---- Fig 3: RPY Angles ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
        ax.plot(t, rpy[:, i], label=name, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('RL RPY Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f'rl_traj_{trajectory_name}_rpy_angles', plot_dir)

    # ---- Fig 4: Pre-allocation Control Signals (PID + RL) ----
    res = log.get('residual')
    has_rl = res is not None and res.shape[0] == pre_alloc.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f'RL Pre-allocation Control Signals (PID + RL Residual): '
        f'{trajectory_name}', fontsize=13)

    att_names = ['Roll', 'Pitch', 'Yaw']
    att_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (name, clr) in enumerate(zip(att_names, att_colors)):
        combined = pre_alloc[:, i] + res[:, i] if has_rl else pre_alloc[:, i]
        axes[0].plot(t, combined, color=clr, alpha=0.8,
                     label=f'{name} (PID+RL)')
        if has_rl:
            axes[0].plot(t, pre_alloc[:, i], color=clr, ls=':', alpha=0.3,
                         lw=0.8, label=f'{name} (PID only)')
    axes[0].set_ylabel('Torque (normalized)')
    axes[0].set_title('Attitude Control Torques  (solid = PID+RL, dotted = PID only)')
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, pre_alloc[:, 3], label='Drive X', alpha=0.8)
    axes[1].plot(t, pre_alloc[:, 4], label='Drive Y', alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Drive (normalized)')
    axes[1].set_title('Drive Acceleration Signals')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, f'rl_traj_{trajectory_name}_pre_alloc_signals', plot_dir)

    _safe_show(plt)


def plot_residual_analysis(log, label):
    """PID vs RL residual + axis-resolved gimbal lock analysis (3-panel).

    Gimbal topology: base→joint1(-Y)→Link1→joint2(tilted)→Link2→joint3(-Y)→shell.
    joint1 & joint3 share the Y axis → when θ₂≈0, they are redundant (lock).

    Which axis is locked depends on θ₁:
      lost_axis_in_world = a₁ × R_joint1 · a₂
      → lost_x = sin(θ₁ - φ)   [Roll lock fraction]
      → lost_z = -cos(θ₁ - φ)  [Yaw lock fraction]
      where φ = atan2(j2z, -j2x) ≈ 32.5° (from joint2 tilt).
      Pitch (Y) is NEVER locked — both joint1/3 provide Y rotation.

    Top:    PID vs RL residual, background: red=Roll locked, blue=Yaw locked
    Middle: Axis-resolved lock severity + per-axis |RL residual| overlay
    Bottom: All 3 joints sin/cos with lock-axis shading
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
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
    pre = log['pre_alloc']    # (N, 5): [ro, po, yo, xd, yd]
    res = log['residual']      # (N, act_dim): [roll, pitch, yaw, ...]
    ja  = log['joint_angles']  # (N, 3): [j1, j2, j3] in radians

    # ── Axis-resolved gimbal lock geometry ──
    phi = np.arctan2(0.53742, 0.84332)  # ≈ 32.5° tilt offset

    lock_severity = 1.0 - np.abs(np.sin(ja[:, 1]))
    roll_lock_frac = np.abs(np.sin(ja[:, 0] - phi))
    yaw_lock_frac  = np.abs(np.cos(ja[:, 0] - phi))

    roll_lock = lock_severity * roll_lock_frac
    yaw_lock  = lock_severity * yaw_lock_frac

    lock_thresh = 0.5
    in_lock = lock_severity > lock_thresh
    roll_dominant = roll_lock_frac > yaw_lock_frac
    roll_lock_mask = in_lock & roll_dominant
    yaw_lock_mask  = in_lock & ~roll_dominant

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(12, 9.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 2]})

    # Unified RGB=XYZ: Roll(X)=Red, Pitch(Y)=Green, Yaw(Z)=Blue
    CLR_ROLL  = '#D32F2F'   # red
    CLR_PITCH = '#2E7D32'   # green
    CLR_YAW   = '#1565C0'   # blue
    axis_colors = [('Roll', CLR_ROLL), ('Pitch', CLR_PITCH), ('Yaw', CLR_YAW)]

    # ── Top: PID attitude + RL residual with lock-axis shading ──
    for j, (name, clr) in enumerate(axis_colors):
        ax_top.plot(t, pre[:, j], color=clr, ls='--', alpha=0.55, lw=1.2,
                    label=f'PID {name}')
        ax_top.plot(t, res[:, j], color=clr, ls='-', alpha=0.85, lw=1.4,
                    label=f'RL Residual {name}')
    ax_top.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax_top.set_ylabel('Control Signal (normalized)')
    ax_top.set_title(f'PID Attitude Corrections vs RL Residual — {label}')

    yl = ax_top.get_ylim()
    if np.any(roll_lock_mask):
        ax_top.fill_between(t, yl[0], yl[1], where=roll_lock_mask,
                            color=CLR_ROLL, alpha=0.10, zorder=0)
    if np.any(yaw_lock_mask):
        ax_top.fill_between(t, yl[0], yl[1], where=yaw_lock_mask,
                            color=CLR_YAW, alpha=0.10, zorder=0)
    ax_top.set_ylim(yl)

    handles, labels = ax_top.get_legend_handles_labels()
    if np.any(roll_lock_mask):
        handles.append(Patch(facecolor=CLR_ROLL, alpha=0.25, edgecolor='none'))
        labels.append('Roll locked')
    if np.any(yaw_lock_mask):
        handles.append(Patch(facecolor=CLR_YAW, alpha=0.25, edgecolor='none'))
        labels.append('Yaw locked')
    ax_top.legend(handles, labels, fontsize=8, ncol=4,
                  loc='upper right', framealpha=0.9, edgecolor='0.8')
    ax_top.grid(True, alpha=0.25, lw=0.5)

    # ── Middle: Axis-resolved lock severity + per-axis RL residual ──
    ax_mid.fill_between(t, 0, roll_lock, color=CLR_ROLL, alpha=0.35,
                        label='Roll lock severity')
    ax_mid.fill_between(t, 0, yaw_lock, color=CLR_YAW, alpha=0.35,
                        label='Yaw lock severity')
    ax_mid.axhline(lock_thresh, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax_mid.set_ylabel('Lock Severity')
    ax_mid.set_ylim(0, 1.05)
    ax_mid.set_title(
        r'Axis-Resolved Gimbal Lock  '
        r'($\theta_1$-dependent direction, Pitch never locked)')

    lock_pct = 100.0 * np.mean(in_lock)
    ax_mid.text(0.02, 0.95, f'In lock: {lock_pct:.0f}% of time',
                transform=ax_mid.transAxes, fontsize=8, va='top', alpha=0.7)

    ax_mid_r = ax_mid.twinx()
    ax_mid_r.plot(t, np.abs(res[:, 0]), color=CLR_ROLL, lw=1.0, alpha=0.55,
                  ls='-', label='|RL Roll residual|')
    ax_mid_r.plot(t, np.abs(res[:, 2]), color=CLR_YAW, lw=1.0, alpha=0.55,
                  ls='-', label='|RL Yaw residual|')
    ax_mid_r.set_ylabel('|RL Residual|', alpha=0.7)
    ax_mid_r.tick_params(axis='y', labelcolor='gray')

    lines1, lab1 = ax_mid.get_legend_handles_labels()
    lines2, lab2 = ax_mid_r.get_legend_handles_labels()
    ax_mid.legend(lines1 + lines2, lab1 + lab2, fontsize=8, ncol=2,
                  loc='upper right', framealpha=0.9, edgecolor='0.8')
    ax_mid.grid(True, alpha=0.25, lw=0.5)

    # ── Bottom: All 3 joints sin/cos with lock-axis shading ──
    j_colors = {'1': '#7570B3', '2': '#D95F02', '3': '#1B9E77'}
    for idx, (jname, clr) in enumerate(j_colors.items()):
        ax_bot.plot(t, np.sin(ja[:, idx]), color=clr, ls='-', alpha=0.85,
                    lw=1.4, label=rf'$\sin(\theta_{jname})$')
        ax_bot.plot(t, np.cos(ja[:, idx]), color=clr, ls='--', alpha=0.5,
                    lw=1.0, label=rf'$\cos(\theta_{jname})$')

    if np.any(roll_lock_mask):
        ax_bot.fill_between(t, -1.15, 1.15, where=roll_lock_mask,
                            color=CLR_ROLL, alpha=0.06, zorder=0)
    if np.any(yaw_lock_mask):
        ax_bot.fill_between(t, -1.15, 1.15, where=yaw_lock_mask,
                            color=CLR_YAW, alpha=0.06, zorder=0)

    ax_bot.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax_bot.set_ylim(-1.15, 1.15)
    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('sin / cos')
    ax_bot.set_title(r'Gimbal Joints — $\theta_1$(Y), $\theta_2$(tilted), '
                     r'$\theta_3$(Y)  [$\theta_1$,$\theta_3$ co-rotate under lock]')
    ax_bot.legend(fontsize=7, ncol=3, loc='upper right',
                  framealpha=0.9, edgecolor='0.8')
    ax_bot.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout()
    _save_fig(fig, f'rl_{label}_residual_analysis', plot_dir)
    _safe_show(plt)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL policy on trajectory tracking")
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'line', 'eight'])
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--line-end-x', type=float, default=5.0)
    parser.add_argument('--line-end-y', type=float, default=0.0)
    parser.add_argument('--max-speed', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=50.0)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--pid-rate', type=float, default=50.0)
    parser.add_argument('--mu-roll', type=float, default=0.01)
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to trained policy checkpoint')
    parser.add_argument('--act-dim', type=int, default=3, choices=[3, 5])
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--unidirectional', action='store_true',
                        help='Unidirectional motor mixing (no reverse)')
    parser.add_argument('--yaw-compensate', action='store_true',
                        help='Rotate drive commands to body frame using yaw')
    args = parser.parse_args()

    if args.policy is None:
        args.policy = str(Path(__file__).parent / 'checkpoints' / 'old1' / 'best.pt')

    # Load RL agent — auto-detect obs_dim from checkpoint
    ckpt_info = peek_checkpoint(args.policy)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
    agent.load(args.policy)
    print(f"Loaded RL policy from {args.policy} (act_dim={args.act_dim}, "
          f"obs_dim={obs_dim}{', no_joint' if no_joint else ''})")

    # Create env (no terrain, flat ground)
    env = SphericalRobotEnv(
        mu_roll=args.mu_roll,
        pid_rate=args.pid_rate,
        alpha_max=args.alpha_max,
        warmup_steps=0,
        max_episode_steps=999999,
        randomize=False,
        act_dim=args.act_dim,
        unidirectional=args.unidirectional,
        no_joint_obs=no_joint,
    )
    env.global_step = 999999  # full alpha
    env.controller.yaw_compensate = args.yaw_compensate

    # Create trajectory and tracker
    traj = make_trajectory(args)
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=args.max_speed)

    # Reset env and settle
    obs = env.reset(target_vx=0.0, target_vy=0.0)

    # Target yaw=0 (world frame); yaw_compensate rotates drive to body frame
    env.controller.set_target(roll=0, pitch=0, yaw=0)
    env.target_yaw = 0.0  # sync for RL obs

    traj_start_time = env.sim.time

    print(f"Running RL {args.trajectory} trajectory "
          f"(speed={args.speed}, duration={args.duration}s)...")

    log = run_eval(env, agent, tracker,
                   args.duration - args.settle, traj_start_time)

    # Summary stats
    error = log['error']
    n = len(error) // 4
    print(f"\nRL Tracking stats (after transient):")
    print(f"  Error: mean={np.mean(error[n:]):.3f}m  "
          f"std={np.std(error[n:]):.3f}m  "
          f"max={np.max(error[n:]):.3f}m")
    rpy = log['rpy']
    for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
        arr = rpy[n:, i]
        print(f"  {name:6s}: mean={np.mean(arr):+.2f}  std={np.std(arr):.2f}  "
              f"max={np.max(np.abs(arr)):.2f}")

    plot_log(log, args.trajectory)
    plot_residual_analysis(log, args.trajectory)


if __name__ == "__main__":
    main()
