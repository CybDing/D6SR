#!/usr/bin/env python3
"""
Benchmark: trajectory comparison figure for the paper benchmark.

Generates a 2-panel PDF figure:
  (a) Without PID Attitude control — drive only, no attitude stabilization
  (b) PID controller and Residual RL agent

Usage:
    mjpython benchmark_trajectory.py
    mjpython benchmark_trajectory.py --trajectory text --text IROS2026
    mjpython benchmark_trajectory.py --radius 10 --speed 0.8 --duration 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from control.trajectory import CircleTrajectory, TextTrajectory, TrajectoryTracker
from rl.env import SphericalRobotEnv, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from control.run_baseline import mix_motors


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_trajectory(args):
    """Create trajectory from CLI args."""
    if args.trajectory == 'text':
        return TextTrajectory(args.text, speed=args.speed)
    return CircleTrajectory(radius=args.radius, speed=args.speed, center=(0, 0))


def _log_stride(args, dt):
    """Use denser logging for text trajectories so glyph strokes stay smooth."""
    sample_dt = 0.05 if args.trajectory == 'text' else 0.20
    return max(1, int(round(sample_dt / dt)))


def _shared_limits(logs, text_mode=False):
    """Shared axis limits make the two benchmark panels directly comparable."""
    pts = np.vstack([np.vstack((log['pos'], log['ref'])) for log in logs])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = np.array([
        max(spans[0] * 0.04, 0.5),
        max(spans[1] * (0.14 if text_mode else 0.08), 0.35),
    ])
    return (mins[0] - pad[0], maxs[0] + pad[0]), (
        mins[1] - pad[1], maxs[1] + pad[1]
    )


# ── Simulation runners ──────────────────────────────────────────────────


def run_pid_only(args, attitude_enabled=True):
    """Run trajectory with PID controller (attitude on/off)."""
    traj = _make_trajectory(args)
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=args.max_speed)

    sim = RobotSimulation(mu_roll=args.mu_roll, force_max=5)
    sim.reset()
    controller = BaselineController(pid_rate=args.pid_rate,
                                    unidirectional=args.unidirectional,
                                    yaw_compensate=args.yaw_compensate)

    # Settle
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = attitude_enabled

    traj_start = sim.time
    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    log_stride = _log_stride(args, dt)
    total = int((args.duration - args.settle) / dt)
    shell_id = sim._shell_body_id
    ctrl = np.zeros(6)

    log = {'pos': [], 'ref': [], 'error': []}

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start
            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)

        if i % log_stride == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - traj_start
            ref_x, ref_y = traj.position(traj_t)
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y))

        if i % int(2.0 / dt) == 0:
            traj_t = sim.time - traj_start
            ball_pos = sim.data.xpos[shell_id][:2]
            err = log['error'][-1] if log['error'] else 0
            label = "PID" if attitude_enabled else "No-PID"
            print(f"\r  [{label}] t={traj_t:5.1f}s  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_rl(args):
    """Run trajectory with PID + RL residual."""
    policy_path = args.policy
    if policy_path is None:
        policy_path = str(Path(__file__).parent / 'rl' / 'checkpoints' / 'old1' / 'best.pt')

    ckpt_info = peek_checkpoint(policy_path)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
    agent.load(policy_path)
    print(f"  Loaded RL policy: {policy_path} (obs_dim={obs_dim})")

    env = SphericalRobotEnv(
        mu_roll=args.mu_roll, pid_rate=args.pid_rate,
        alpha_max=args.alpha_max, warmup_steps=0,
        max_episode_steps=999999, randomize=False,
        act_dim=args.act_dim, unidirectional=args.unidirectional,
        no_joint_obs=no_joint,
    )
    env.global_step = 999999  # full alpha
    env.controller.yaw_compensate = args.yaw_compensate

    traj = _make_trajectory(args)
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=args.max_speed)

    obs = env.reset(target_vx=0.0, target_vy=0.0)
    env.controller.set_target(roll=0, pitch=0, yaw=0)
    env.target_yaw = 0.0

    traj_start = env.sim.time
    dt = env.sim.dt
    steps_per_pid = max(1, int(round(1.0 / env.pid_rate / dt)))
    log_stride = _log_stride(args, dt)
    total = int((args.duration - args.settle) / dt)
    shell_id = env.shell_id
    controller = env.controller

    log = {'pos': [], 'ref': [], 'error': []}
    residual = np.zeros(env.act_dim)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = env.sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(env.sim.model, env.sim.data, shell_id)
            traj_t = env.sim.time - traj_start

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            obs = env._get_obs()
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

            imu = env.sim.get_imu()
            pid_ctrl = controller.update(imu, ball_vel)

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

        if i % log_stride == 0:
            ball_pos = env.sim.data.xpos[shell_id][:2].copy()
            traj_t = env.sim.time - traj_start
            ref_x, ref_y = traj.position(traj_t)
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y))

        if i % int(2.0 / dt) == 0:
            traj_t = env.sim.time - traj_start
            ball_pos = env.sim.data.xpos[shell_id][:2]
            err = log['error'][-1] if log['error'] else 0
            print(f"\r  [RL] t={traj_t:5.1f}s  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        env.sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_comparison(log_nopid, log_rl, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    is_text = hasattr(args, 'trajectory') and args.trajectory == 'text'
    if is_text:
        fig, (ax_a, ax_b) = plt.subplots(
            2, 1, figsize=(12.0, 6.4), sharex=True, sharey=True
        )
    else:
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(7.16, 3.5), sharex=True, sharey=True
        )

    xlim, ylim = _shared_limits([log_nopid, log_rl], text_mode=is_text)

    # ── Panel (a): No PID attitude ──
    _plot_panel(
        ax_a, log_nopid, '(a) Without PID Attitude control', 'Actual',
        xlim=xlim, ylim=ylim, text_mode=is_text
    )

    # ── Panel (b): PID + RL ──
    _plot_panel(
        ax_b, log_rl, '(b) PID controller and Residual RL agent', 'Actual (RL)',
        xlim=xlim, ylim=ylim, text_mode=is_text
    )

    if is_text:
        ax_a.set_xlabel('')
        ax_a.tick_params(labelbottom=False)
        fig.tight_layout(h_pad=1.1)
    else:
        fig.tight_layout(w_pad=1.5)

    out_dir = Path(__file__).parent / 'plots'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'benchmark_trajectory_comparison.pdf'
    fig.savefig(str(out_path), format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Also print tracking stats
    for label, log in [("No-PID", log_nopid), ("PID+RL", log_rl)]:
        err = log['error']
        n = len(err) // 4
        print(f"  {label}: mean_err={np.mean(err[n:]):.3f}m  "
              f"max_err={np.max(err[n:]):.3f}m")


def _plot_panel(ax, log, title, actual_label, xlim=None, ylim=None,
                text_mode=False):
    pos = log['pos']
    ref = log['ref']

    # Reference trajectory
    ax.plot(ref[:, 0], ref[:, 1], color='0.55', ls=(0, (4, 2)), alpha=0.75,
            label='Reference', linewidth=1.3, dash_capstyle='round',
            solid_joinstyle='round')

    # Actual trajectory
    ax.plot(pos[:, 0], pos[:, 1], color='#1f77b4', alpha=0.9,
            label=actual_label, linewidth=1.8, solid_capstyle='round',
            solid_joinstyle='round')

    # Start / End markers
    ax.plot(pos[0, 0], pos[0, 1], 'o', color='#2ca02c', markersize=7,
            markeredgecolor='white', markeredgewidth=0.5, label='Start',
            zorder=5)
    ax.plot(pos[-1, 0], pos[-1, 1], 's', color='#ff7f0e', markersize=7,
            markeredgecolor='white', markeredgewidth=0.5, label='End',
            zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    legend_kwargs = {
        'framealpha': 0.92,
        'edgecolor': '0.8',
        'handlelength': 1.8,
        'borderpad': 0.4,
        'labelspacing': 0.3,
    }
    if text_mode:
        legend_kwargs.update({
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, 1.18),
            'ncol': 4,
            'columnspacing': 1.1,
        })
    else:
        legend_kwargs['loc'] = 'upper right'
    ax.legend(**legend_kwargs)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: trajectory comparison figure for IEEE paper")
    parser.add_argument('--trajectory', default='text',
                        choices=['circle', 'text'],
                        help='Trajectory type (default: text)')
    parser.add_argument('--text', type=str, default='IROS2026',
                        help='Text to trace (only for --trajectory text)')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Circle radius in meters (only for --trajectory circle)')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (auto-computed from trajectory if omitted)')
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--pid-rate', type=float, default=50.0)
    parser.add_argument('--mu-roll', type=float, default=0.01)
    parser.add_argument('--max-speed', type=float, default=1.0)
    parser.add_argument('--unidirectional', action='store_true', default=True)
    parser.add_argument('--yaw-compensate', action='store_true', default=True)
    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('--act-dim', type=int, default=3)
    parser.add_argument('--alpha-max', type=float, default=0.15)
    args = parser.parse_args()

    # Auto-compute duration from trajectory if not specified
    if args.duration is None:
        tmp_traj = _make_trajectory(args)
        traj_dur = tmp_traj.duration()
        if traj_dur == float('inf'):
            args.duration = 200.0  # default for periodic trajectories
        else:
            args.duration = traj_dur + args.settle + 10.0

    print("=" * 60)
    print("Benchmark: Trajectory Comparison")
    print(f"  trajectory={args.trajectory}  "
          f"speed={args.speed}  duration={args.duration:.1f}s")
    if args.trajectory == 'text':
        print(f"  text='{args.text}'")
    else:
        print(f"  radius={args.radius}")
    print("=" * 60)

    # (a) Without PID attitude control
    print("\n[1/2] Running WITHOUT PID attitude control...")
    log_nopid = run_pid_only(args, attitude_enabled=False)

    # (b) PID + RL
    print("\n[2/2] Running PID + Residual RL agent...")
    log_rl = run_rl(args)

    # Plot
    print("\nGenerating IEEE figure...")
    plot_comparison(log_nopid, log_rl, args)


if __name__ == "__main__":
    main()
