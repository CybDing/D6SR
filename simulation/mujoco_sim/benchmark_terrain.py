#!/usr/bin/env python3
"""
Multi-seed terrain benchmark: compare No-PID, PID-only, and PID+RL conditions
on rough terrain with trajectory tracking.

Runs N seeds × 3 conditions, each for exactly --duration seconds (settle excluded).
Plots mean Yaw ± 1σ across seeds, matching the paper figure style.

Usage:
    mjpython benchmark_terrain.py --seeds 0 1 2 3 4 --roughness 0.2 --headless --plot
    mjpython benchmark_terrain.py --seeds 0 1 --roughness 0.2 --headless --plot --no-rl
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from control.run_terrain_test import run_headless_traj
from control.trajectory import LineTrajectory, TrajectoryTracker
from terrain import RoughTerrain


# ---------------------------------------------------------------------------
# Trial helpers
# ---------------------------------------------------------------------------

def _setup(seed, args):
    """Create terrain+sim, settle, return (sim, controller, imu_after_settle)."""
    terrain = RoughTerrain(
        roughness=args.roughness, seed=seed,
        nrow=args.hfield_res, ncol=args.hfield_res,
        size_x=args.hfield_size, size_y=args.hfield_size,
    )
    sim = RobotSimulation(mu_roll=args.mu_roll, terrain=terrain, force_max=5)
    sim.reset()

    # Settle phase — NOT counted in the 50 s
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    imu = sim.get_imu()
    controller = BaselineController(pid_rate=args.pid_rate)
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    return sim, controller, imu


def _make_tracker(args):
    """Fresh LineTrajectory + TrajectoryTracker."""
    angle_rad = np.radians(args.line_angle)
    end_x = args.line_length * np.cos(angle_rad)
    end_y = args.line_length * np.sin(angle_rad)
    traj = LineTrajectory(start=(0, 0), end=(end_x, end_y), speed=args.speed)
    return TrajectoryTracker(traj, dt=1.0 / args.pid_rate)


def run_trial_no_pid(seed, args):
    """Condition (a): drive-only, no attitude PID."""
    print(f"  [no_pid] seed={seed}")
    sim, controller, _ = _setup(seed, args)
    controller.enabled = False          # disable roll/pitch/yaw; drive PID still runs
    tracker = _make_tracker(args)
    traj_start_time = sim.time
    return run_headless_traj(sim, controller, tracker, args.duration, traj_start_time)


def run_trial_pid(seed, args):
    """Condition (b): full PID attitude control + trajectory tracking."""
    print(f"  [pid]    seed={seed}")
    sim, controller, _ = _setup(seed, args)
    controller.enabled = True
    tracker = _make_tracker(args)
    traj_start_time = sim.time
    return run_headless_traj(sim, controller, tracker, args.duration, traj_start_time)


def run_trial_pid_rl(seed, args, agent, env):
    """Condition (c): PID + RL residual.

    Inline loop — identical structure to run_headless_traj but with RL residual
    injected after PID update. No extra pre-stabilisation (fair comparison).
    """
    print(f"  [pid_rl] seed={seed}")
    sim, controller, imu = _setup(seed, args)
    controller.enabled = True
    tracker = _make_tracker(args)

    # Point env at this trial's sim and controller
    env.sim = sim
    env.shell_id = sim._shell_body_id
    env.controller = controller
    env.target_yaw = np.radians(imu['euler'][2])
    env.global_step = int(1e9)          # alpha = alpha_max (fully trained)

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / args.pid_rate / dt)))
    total = int(args.duration / dt)
    shell_id = sim._shell_body_id
    traj_start_time = sim.time

    log = {'t': [], 'rpy': [], 'ctrl': [], 'pos': [],
           'pre_alloc': [], 'ref': [], 'error': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - traj_start_time

            vx, vy, ref_x, ref_y = tracker.compute(traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            pid_ctrl = controller.update(sim.get_imu(), ball_vel)

            obs = env._get_obs()
            action = np.clip(
                np.asarray(agent.select_action(obs, deterministic=True),
                           dtype=np.float64),
                -1.0, 1.0)
            residual_6 = env._mix_residual(args.alpha_max * action)
            ctrl = np.clip(pid_ctrl + residual_6, -1.0, 1.0)
            sim.set_ctrl(ctrl)

        if i % 20 == 0:
            imu_now = sim.get_imu()
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - traj_start_time
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['rpy'].append(imu_now['euler'].copy())
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
                  f"err={err:.3f}m", end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_logs, duration, n_pts=2500):
    """Interpolate per-seed logs to a common time grid; compute mean ± std.

    Returns dict with keys:
        't'    (n_pts,)   — common time axis
        'mean' (n_pts, 3) — mean roll/pitch/yaw across seeds
        'std'  (n_pts, 3) — std  roll/pitch/yaw across seeds
    """
    t_common = np.linspace(0.0, duration, n_pts)
    rpy_all = []

    for log in seed_logs:
        t = log['t']
        rpy = log['rpy'].copy()
        # Unwrap each angle to avoid interpolation artefacts at ±180° wrap
        for i in range(3):
            rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))
        # Clip t_common to the actual range of this log (handles slight length diffs)
        t_clipped = np.clip(t_common, t[0], t[-1])
        rpy_interp = np.stack(
            [np.interp(t_clipped, t, rpy[:, i]) for i in range(3)],
            axis=1)
        rpy_all.append(rpy_interp)

    rpy_stack = np.stack(rpy_all, axis=0)   # (N_seeds, n_pts, 3)
    return {
        't':    t_common,
        'mean': rpy_stack.mean(axis=0),
        'std':  rpy_stack.std(axis=0),
        'n':    len(seed_logs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_show(plt):
    backend = plt.get_backend().lower()
    if 'agg' not in backend:
        try:
            plt.show()
        except Exception:
            pass


def plot_benchmark(results, args):
    """3-panel Yaw stability plot, one panel per condition, mean +/- 1 sigma bands.

    IEEE-compliant: Times New Roman, 7.16-inch width, PDF output, inward ticks.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
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

    COND_COLORS = {
        'no_pid':  '#d62728',   # red
        'pid':     '#1f77b4',   # blue
        'pid_rl':  '#2ca02c',   # green
    }
    ALPHA_FILL = 0.15
    ALPHA_LINE = 0.9

    COND_ORDER = ['no_pid', 'pid', 'pid_rl']
    PANEL_LABELS = {
        'no_pid':  'Without Attitude control',
        'pid':     'PID Controller only',
        'pid_rl':  'PID controller and Residual RL agent',
    }

    conditions = [c for c in COND_ORDER if c in results]
    n = len(conditions)

    fig, axes = plt.subplots(n, 1, figsize=(7.16, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax_i, cond in enumerate(conditions):
        agg = results[cond]
        t = agg['t']
        yaw_mean = agg['mean'][:, 2]
        yaw_std  = agg['std'][:, 2]
        color = COND_COLORS.get(cond, '#2ca02c')

        grand_mean = float(np.mean(yaw_mean))
        grand_std  = float(np.mean(yaw_std))
        max_abs    = float(np.max(np.abs(yaw_mean) + yaw_std))

        env_lo = float(np.min(yaw_mean - yaw_std))
        env_hi = float(np.max(yaw_mean + yaw_std))
        margin = max((env_hi - env_lo) * 0.20, 2.0)
        ax = axes[ax_i]
        ax.set_ylim(env_lo - margin, env_hi + margin)

        # Mean line
        ax.plot(t, yaw_mean, color=color, alpha=ALPHA_LINE, linewidth=1.5,
                label=r'Mean Yaw ($\mu$)')
        # +/- 1 sigma band
        ax.fill_between(t, yaw_mean - yaw_std, yaw_mean + yaw_std,
                        color=color, alpha=ALPHA_FILL,
                        label=r'$\pm 1\sigma$')
        # Dashed time-mean line
        ax.axhline(grand_mean, color=color, ls='--', alpha=0.45, linewidth=1.0)

        # Panel label
        letter = chr(ord('a') + ax_i)
        ax.text(0.01, 0.95,
                f'({letter}) {PANEL_LABELS[cond]}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, fontweight='bold')

        # Stats annotation (top-right)
        stats_txt = (f'mean={grand_mean:+.1f}\u00b0, '
                     f'std={grand_std:.1f}\u00b0, '
                     f'max={max_abs:.1f}\u00b0')
        ax.text(0.99, 0.95, stats_txt,
                transform=ax.transAxes, va='top', ha='right',
                fontsize=8, color='0.35')

        ax.set_ylabel('Yaw (\u00b0)')
        ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.78),
                  framealpha=0.9, edgecolor='0.8',
                  handlelength=1.5, borderpad=0.3, labelspacing=0.25)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, args.duration)

    fig.tight_layout(h_pad=0.6)

    # Save as PDF
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    seeds_str = '-'.join(str(s) for s in sorted(args.seeds))
    tag = f"rough{args.roughness}_seeds{seeds_str}"
    out_pdf = plot_dir / f'benchmark_terrain_{tag}.pdf'
    fig.savefig(str(out_pdf), format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    # Also save PNG for quick preview
    out_png = plot_dir / f'benchmark_terrain_{tag}.png'
    fig.savefig(str(out_png), dpi=200, bbox_inches='tight')
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed terrain benchmark: No-PID vs PID vs PID+RL')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Random seeds for terrain generation (default: 0 1 2 3 4)')
    parser.add_argument('--roughness', type=float, default=0.2,
                        help='RoughTerrain roughness in metres (default: 0.2)')
    parser.add_argument('--duration', type=float, default=50.0,
                        help='Trajectory tracking duration in seconds, '
                             'excludes settle (default: 50.0)')
    parser.add_argument('--settle', type=float, default=0.5,
                        help='Settle time before tracking starts (default: 0.5 s)')
    parser.add_argument('--speed', type=float, default=0.5,
                        help='Trajectory speed (default: 0.5 m/s)')
    parser.add_argument('--line-length', type=float, default=25.0,
                        help='Trajectory line length (default: 25.0 m)')
    parser.add_argument('--line-angle', type=float, default=0.0,
                        help='Trajectory direction, degrees from +X (default: 0)')
    parser.add_argument('--pid-rate', type=float, default=50.0,
                        help='PID update rate in Hz (default: 50.0)')
    parser.add_argument('--mu-roll', type=float, default=0.01,
                        help='Rolling friction coefficient (default: 0.01)')
    parser.add_argument('--hfield-res', type=int, default=600,
                        help='Heightfield resolution (default: 600)')
    parser.add_argument('--hfield-size', type=float, default=100.0,
                        help='Heightfield half-extent in metres (default: 100.0)')
    # RL options
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to RL checkpoint '
                             '(default: rl/checkpoints/best.pt)')
    parser.add_argument('--alpha-max', type=float, default=0.15,
                        help='RL residual scale factor (default: 0.15)')
    parser.add_argument('--act-dim', type=int, default=3,
                        help='RL action dimension, 3 or 5 (default: 3)')
    parser.add_argument('--no-rl', action='store_true',
                        help='Skip PID+RL condition (2-condition benchmark)')
    # Output
    parser.add_argument('--headless', action='store_true',
                        help='Run without viewer (auto-set when --plot)')
    parser.add_argument('--plot', action='store_true',
                        help='Save figure to plots/ (PDF + PNG)')
    args = parser.parse_args()

    if args.plot:
        args.headless = True

    # ------------------------------------------------------------------
    # RL setup (once, reuse across seeds)
    # ------------------------------------------------------------------
    agent = None
    env = None
    if not args.no_rl:
        from rl.sac import SAC, peek_checkpoint
        from rl.env import SphericalRobotEnv, OBS_DIM_FULL, OBS_DIM_NO_JOINT

        policy_path = args.policy or str(
            Path(__file__).parent / 'rl' / 'checkpoints' / 'old1' / 'best.pt')
        print(f"Loading RL policy: {policy_path}")
        ckpt_info = peek_checkpoint(policy_path)
        obs_dim = ckpt_info['obs_dim']
        no_joint = (obs_dim == OBS_DIM_NO_JOINT)
        agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
        agent.load(policy_path)
        # Create a minimal env shell; sim/controller replaced per trial
        env = SphericalRobotEnv(alpha_max=args.alpha_max, act_dim=args.act_dim,
                                no_joint_obs=no_joint)

    n_cond = 2 + (0 if args.no_rl else 1)
    print(f"\n{'='*60}")
    print(f"Terrain Benchmark: {len(args.seeds)} seeds × {n_cond} conditions")
    print(f"  Roughness : {args.roughness} m")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Duration  : {args.duration} s  (+ {args.settle} s settle)")
    print(f"  Trajectory: length={args.line_length} m, speed={args.speed} m/s, "
          f"angle={args.line_angle}°")
    print(f"{'='*60}\n")

    results = {}

    # ------------------------------------------------------------------
    # Condition (a): No attitude PID
    # ------------------------------------------------------------------
    print("=== (a) No PID attitude control ===")
    logs = []
    for seed in args.seeds:
        logs.append(run_trial_no_pid(seed, args))
    results['no_pid'] = aggregate_seeds(logs, args.duration)

    # ------------------------------------------------------------------
    # Condition (b): PID only
    # ------------------------------------------------------------------
    print("\n=== (b) PID Controller only ===")
    logs = []
    for seed in args.seeds:
        logs.append(run_trial_pid(seed, args))
    results['pid'] = aggregate_seeds(logs, args.duration)

    # ------------------------------------------------------------------
    # Condition (c): PID + Residual RL
    # ------------------------------------------------------------------
    if not args.no_rl:
        print("\n=== (c) PID + Residual RL ===")
        logs = []
        for seed in args.seeds:
            logs.append(run_trial_pid_rl(seed, args, agent, env))
        results['pid_rl'] = aggregate_seeds(logs, args.duration)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    NAMES = {'no_pid': 'No PID  ', 'pid': 'PID only', 'pid_rl': 'PID+RL  '}
    print(f"\n{'='*60}")
    print("Summary — Yaw axis (mean curve statistics)")
    print(f"{'='*60}")
    for cond, agg in results.items():
        yaw_mean = agg['mean'][:, 2]
        yaw_std  = agg['std'][:, 2]
        print(f"  {NAMES.get(cond, cond)}:  "
              f"grand_mean={np.mean(yaw_mean):+8.2f}°  "
              f"avg_seed_std={np.mean(yaw_std):7.2f}°  "
              f"max|mean_yaw|={np.max(np.abs(yaw_mean)):7.2f}°")

    if args.plot:
        print("\nGenerating plot...")
        plot_benchmark(results, args)

    return results


if __name__ == '__main__':
    main()
