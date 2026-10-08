#!/usr/bin/env python3
"""
Benchmark: V0 (shell-fixed PID) vs V1 (gimbal PID / PID+RL) motor allocation.

(a) V0: PID velocity control with rotation-compensated allocation
(b) V1: PID baseline (bidirectional or unidirectional)
(c) V1: PID baseline + SAC residual correction (optional)

Usage:
    python benchmark_baseline.py                          # V0 + V1-PID + V1-RL
    python benchmark_baseline.py --no-rl                  # V0 + V1-PID only
    python benchmark_baseline.py --unidirectional         # add unidirectional panel
    python benchmark_baseline.py --no-rl --unidirectional # V0 + V1-uni + V1-bid
    python benchmark_baseline.py --skip-v0 --no-rl       # V1-PID only
    python benchmark_baseline.py --policy rl/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from paper_style import PAPER_RC, panel_caption


# ── V0: PID with rotation-compensated allocation ─────────────────────

def collect_run_v0(drive_x, drive_y, duration, settle,
                   pid_rate=50.0, mu_roll=0.01, force_max=5):
    """Run V0 robot (motors fixed on shell) with PID velocity control."""
    mjcf = str(Path(__file__).parent / 'models' / 'robot_v0_mujoco.xml')
    sim = RobotSimulation(mjcf_path=mjcf, mu_roll=mu_roll, force_max=force_max)
    sim.reset()

    controller = BaselineController(
        pid_rate=pid_rate, v0_mode=True,
        motor_radius=sim._motor_radius, force_scale=2.0,
    )

    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    shell_id = sim._shell_body_id

    for _ in range(int(settle / dt)):
        sim.step()

    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.set_drive(vx=drive_x, vy=drive_y)
    controller.enabled = True

    total = int(duration / dt)
    log_interval = max(1, int(round(0.001 / dt)))
    log = {'t': [], 'ctrl': [], 'rpy': [], 'pos': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)
        if i % log_interval == 0:
            log['t'].append(sim.time - settle)
            log['ctrl'].append(ctrl.copy())
            log['rpy'].append(sim.get_imu()['euler'].copy())
            log['pos'].append(sim.data.xpos[shell_id].copy())
        sim.step()

    pos = sim.data.xpos[shell_id]
    print(f"  V0 (PID): travel=({pos[0]:+.2f}, {pos[1]:+.2f}) m")
    return {k: np.array(v) for k, v in log.items()}


# ── V1: PID only (no RL) ─────────────────────────────────────────────

def collect_run_v1(drive_x, drive_y, duration, settle,
                   pid_rate=50.0, mu_roll=0.01, force_max=5,
                   unidirectional=False):
    """Run V1 robot (gimbal) with PID velocity control only (no RL residual)."""
    mjcf = str(Path(__file__).parent / 'models' / 'robot_v1_mujoco.xml')
    sim = RobotSimulation(mjcf_path=mjcf, mu_roll=mu_roll, force_max=force_max)
    sim.reset()

    controller = BaselineController(
        pid_rate=pid_rate, v0_mode=False,
        motor_radius=sim._motor_radius, force_scale=2.0,
        unidirectional=unidirectional,
    )

    dt = sim.dt
    steps_per_pid = max(1, int(round(controller.pid_dt / dt)))
    shell_id = sim._shell_body_id

    for _ in range(int(settle / dt)):
        sim.step()

    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.set_drive(vx=drive_x, vy=drive_y)
    controller.enabled = True

    total = int(duration / dt)
    log_interval = max(1, int(round(0.001 / dt)))
    log = {'t': [], 'ctrl': [], 'rpy': [], 'pos': []}
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)
        if i % log_interval == 0:
            log['t'].append(sim.time - settle)
            log['ctrl'].append(ctrl.copy())
            log['rpy'].append(sim.get_imu()['euler'].copy())
            log['pos'].append(sim.data.xpos[shell_id].copy())
        sim.step()

    uni_str = ", unidirectional" if unidirectional else ""
    pos = sim.data.xpos[shell_id]
    print(f"  V1 (PID{uni_str}): travel=({pos[0]:+.2f}, {pos[1]:+.2f}) m")
    return {k: np.array(v) for k, v in log.items()}


# ── V1: PID + SAC residual ───────────────────────────────────────────

def collect_run_rl(policy_path, drive_x, drive_y, duration, settle,
                   pid_rate=50.0, mu_roll=0.01, force_max=5):
    """Run V1 robot with PID baseline + SAC residual from checkpoint."""
    from rl.sac import SAC, peek_checkpoint
    from rl.env import SphericalRobotEnv, OBS_DIM_NO_JOINT

    # Load agent
    ckpt_info = peek_checkpoint(policy_path)
    obs_dim = ckpt_info['obs_dim']
    act_dim = ckpt_info['act_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    print(f"  Loading SAC: obs_dim={obs_dim}, act_dim={act_dim}, "
          f"no_joint={no_joint}, hidden={ckpt_info['hidden']}")

    agent = SAC(obs_dim=obs_dim, act_dim=act_dim)
    agent.load(policy_path)

    # Create env (V1 with gimbal, flat terrain for benchmark)
    env = SphericalRobotEnv(
        force_max=force_max, mu_roll=mu_roll, pid_rate=pid_rate,
        alpha_max=0.15, warmup_steps=0,   # full alpha immediately
        max_episode_steps=999999,
        settle_seconds=settle,
        randomize=False,                  # deterministic for benchmark
        act_dim=act_dim,
        terrain_prob=0.0,                 # flat ground
        no_joint_obs=no_joint,
    )
    # Force global_step high so alpha = alpha_max
    env.global_step = env.warmup_steps + 1

    obs = env.reset(target_vx=drive_x, target_vy=drive_y)
    shell_id = env.shell_id
    dt = env.dt_physics
    steps_per_pid = env.steps_per_pid

    total_pid_steps = int(duration / env.pid_dt)
    log_phys_interval = max(1, int(round(0.001 / dt)))
    log = {'t': [], 'ctrl': [], 'rpy': [], 'pos': []}

    for step in range(total_pid_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Log at ~1ms resolution within each PID step
        ctrl_total = env.sim.data.ctrl.copy()
        t_now = env.sim.time - settle
        imu = env.sim.get_imu()
        log['t'].append(t_now)
        log['ctrl'].append(ctrl_total)
        log['rpy'].append(imu['euler'].copy())
        log['pos'].append(env.sim.data.xpos[shell_id].copy())

        if done:
            print(f"  Episode ended at step {step}")
            break

    pos = env.sim.data.xpos[shell_id]
    print(f"  V1 (PID+RL): travel=({pos[0]:+.2f}, {pos[1]:+.2f}) m, "
          f"alpha={env.alpha:.3f}")
    return {k: np.array(v) for k, v in log.items()}


# ── IEEE-style plot ───────────────────────────────────────────────────

def plot_comparison(panels_data, drive_x, drive_y, out_path):
    """Plot motor allocation comparison.

    Parameters
    ----------
    panels_data : list of (log_dict, label_str)
        Each entry is a (log, label) pair to render as a subplot panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        **PAPER_RC,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.8,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.75',
        'legend.fancybox': False,
    })

    pair_colors = ['#1f77b4', '#d62728', '#2ca02c']
    pair_labels = [r'\theta_y', r'\theta_r', r'\theta_p']

    n_panels = len(panels_data)
    # each panel reserves room under it for its caption
    fig_h = 1.85 * n_panels + 0.4
    fig, axes = plt.subplots(n_panels, 1, figsize=(3.45, fig_h), sharex=True,
                             gridspec_kw={'hspace': 0.32})
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)

    drive_label = []
    if drive_x: drive_label.append(f'$v_x={drive_x}$')
    if drive_y: drive_label.append(f'$v_y={drive_y}$')
    drive_str = ', '.join(drive_label) + ' m/s'

    for ax, (log, label) in zip(axes, panels_data):
        t = log['t']
        ctrl = log['ctrl']
        for pair_idx in range(3):
            c = pair_colors[pair_idx]
            i_neg, i_pos = pair_idx * 2, pair_idx * 2 + 1
            ax.plot(t, ctrl[:, i_neg], ls='-',  color=c, linewidth=0.85,
                    label=f'$M_{{{i_neg}}}$ (${pair_labels[pair_idx]}^-$)')
            ax.plot(t, ctrl[:, i_pos], ls='--', color=c, linewidth=0.85,
                    label=f'$M_{{{i_pos}}}$ (${pair_labels[pair_idx]}^+$)')
        ax.set_ylabel('Normalized thrust')
        ax.grid(True)
        ax.set_ylim(-1.08, 1.08)
        ax.axhline(0, color='k', linewidth=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center',
               bbox_to_anchor=(0.55, 1.0),
               columnspacing=0.7, handlelength=1.6, handletextpad=0.3,
               borderpad=0.25, fontsize=6.5)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel('Time (s)')

    for ax, (_, label) in zip(axes, panels_data):
        panel_caption(ax, label)

    axes[-1].text(0.98, 0.05, drive_str, transform=axes[-1].transAxes,
                  fontsize=7, ha='right', va='bottom', style='italic',
                  color='0.4')

    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved → {out_path}")


def generate_gif(panels_data, drive_x, drive_y, out_path, fps=20, duration=10.0,
                  gif_duration=4.2):
    """Generate animated GIF showing motor allocation building over time.

    Each frame reveals data from t=0 up to the current time, so the viewer
    can see how propeller speeds evolve during rolling.

    gif_duration: target playback length in seconds (speedup = duration / gif_duration).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import imageio.v2 as imageio

    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.8,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.75',
        'legend.fancybox': False,
        'mathtext.fontset': 'cm',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    pair_colors = ['#1f77b4', '#d62728', '#2ca02c']
    pair_labels = [r'\theta_y', r'\theta_r', r'\theta_p']

    n_panels = len(panels_data)
    fig_h = 1.6 * n_panels + 0.4
    total_frames = int(fps * gif_duration)
    speedup = duration / gif_duration

    drive_label = []
    if drive_x: drive_label.append(f'$v_x={drive_x}$')
    if drive_y: drive_label.append(f'$v_y={drive_y}$')
    drive_str = ', '.join(drive_label) + ' m/s'

    writer = imageio.get_writer(str(out_path), mode='I', fps=fps, loop=0)

    for f_idx in range(total_frames + 1):
        t_current = f_idx / fps * speedup

        fig, axes = plt.subplots(n_panels, 1, figsize=(6.9, fig_h * 2),
                                 sharex=True, gridspec_kw={'hspace': 0.08})
        if n_panels == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10)

        for ax, (log, label) in zip(axes, panels_data):
            t = log['t']
            ctrl = log['ctrl']
            mask = t <= t_current
            t_vis = t[mask]
            ctrl_vis = ctrl[mask]

            for pair_idx in range(3):
                c = pair_colors[pair_idx]
                i_neg, i_pos = pair_idx * 2, pair_idx * 2 + 1
                lbl_neg = f'$M_{{{i_neg}}}$ (${pair_labels[pair_idx]}^-$)'
                lbl_pos = f'$M_{{{i_pos}}}$ (${pair_labels[pair_idx]}^+$)'
                if len(t_vis) > 0:
                    ax.plot(t_vis, ctrl_vis[:, i_neg], ls='-', color=c,
                            linewidth=0.85, label=lbl_neg)
                    ax.plot(t_vis, ctrl_vis[:, i_pos], ls='--', color=c,
                            linewidth=0.85, label=lbl_pos)
                else:
                    ax.plot([], [], ls='-', color=c, linewidth=0.85, label=lbl_neg)
                    ax.plot([], [], ls='--', color=c, linewidth=0.85, label=lbl_pos)

            ax.set_ylabel('Normalized thrust')
            ax.text(0.02, 0.97, label, transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top',
                    bbox=dict(boxstyle='square,pad=0.12', fc='white',
                              ec='0.7', lw=0.4, alpha=0.92))
            ax.grid(True)
            ax.set_ylim(-1.08, 1.08)
            ax.set_xlim(0, duration)
            ax.axhline(0, color='k', linewidth=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='upper center',
                   bbox_to_anchor=(0.55, 1.0),
                   columnspacing=0.7, handlelength=1.6, handletextpad=0.3,
                   borderpad=0.25, fontsize=6.5)

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel('Time (s)')
        axes[-1].text(0.98, 0.05, drive_str, transform=axes[-1].transAxes,
                      fontsize=7, ha='right', va='bottom', style='italic',
                      color='0.4')
        axes[-1].text(0.98, 0.95, f't = {t_current:.1f} s',
                      transform=axes[-1].transAxes,
                      fontsize=9, ha='right', va='top', fontweight='bold',
                      color='0.3')

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)
        plt.close(fig)

        if f_idx % 20 == 0 or f_idx == total_frames:
            print(f"\r  GIF frame {f_idx}/{total_frames} "
                  f"(t={t_current:.1f}s)", end="", flush=True)

    writer.close()
    print(f"\nSaved GIF → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark: V0 (PID) vs V1 (PID/PID+RL) motor allocation')
    parser.add_argument('--drive-x', type=float, default=1.5)
    parser.add_argument('--drive-y', type=float, default=0.0)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--force-max', type=float, default=5)
    parser.add_argument('--policy', type=str, default=None,
                        help='SAC checkpoint path (default: rl/checkpoints/old1/best.pt)')
    parser.add_argument('--no-rl', action='store_true',
                        help='Skip residual RL policy, run V1 with PID only')
    parser.add_argument('--unidirectional', action='store_true',
                        help='Use unidirectional motor mixing for V1')
    parser.add_argument('--skip-v0', action='store_true',
                        help='Skip V0 (shell-fixed) run')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--gif', action='store_true',
                        help='Generate animated GIF of motor allocation over time')
    parser.add_argument('--gif-fps', type=int, default=20,
                        help='GIF frame rate (default: 20)')
    args = parser.parse_args()

    if args.policy is None:
        args.policy = str(Path(__file__).parent / 'rl' / 'checkpoints' / 'old1' /'best.pt')
    out = Path(args.output) if args.output else \
        Path(__file__).parent / 'benchmark_motor_allocation.pdf'

    panels = []
    panel_letter = ord('a')

    if not args.skip_v0:
        print("Running V0 (shell-fixed, PID only)...")
        log_v0 = collect_run_v0(
            args.drive_x, args.drive_y, args.duration, args.settle,
            force_max=args.force_max)
        panels.append((log_v0, f'({chr(panel_letter)}) S. Sabet et al. [9]'))
        panel_letter += 1

    if args.unidirectional:
        print("Running V1 (gimbal, PID, unidirectional)...")
        log_v1_uni = collect_run_v1(
            args.drive_x, args.drive_y, args.duration, args.settle,
            force_max=args.force_max, unidirectional=True)
        panels.append((log_v1_uni, f'({chr(panel_letter)}) Ours (unidirectional)'))
        panel_letter += 1

    if not args.no_rl:
        # Also run bidirectional PID-only for comparison when RL is included
        print("Running V1 (gimbal, PID only, bidirectional)...")
        log_v1_pid = collect_run_v1(
            args.drive_x, args.drive_y, args.duration, args.settle,
            force_max=args.force_max, unidirectional=False)
        panels.append((log_v1_pid, f'({chr(panel_letter)}) Ours (PID only)'))
        panel_letter += 1

        print("Running V1 (gimbal, PID + SAC residual)...")
        log_v1_rl = collect_run_rl(
            args.policy, args.drive_x, args.drive_y, args.duration, args.settle,
            force_max=args.force_max)
        panels.append((log_v1_rl, f'({chr(panel_letter)}) Ours (PID + RL)'))
        panel_letter += 1
    else:
        print("Running V1 (gimbal, PID only, bidirectional)...")
        log_v1_pid = collect_run_v1(
            args.drive_x, args.drive_y, args.duration, args.settle,
            force_max=args.force_max, unidirectional=False)
        panels.append((log_v1_pid, f'({chr(panel_letter)}) Ours'))
        panel_letter += 1

    if not panels:
        print("No runs selected. Use fewer --skip flags.")
        return

    plot_comparison(panels, args.drive_x, args.drive_y, out)

    if args.gif:
        for i, (log, label) in enumerate(panels):
            tag = label.split(')')[-1].strip().lower().replace(' ', '_')
            gif_path = out.with_name(f'benchmark_motor_{tag}.gif')
            generate_gif([(log, label)], args.drive_x, args.drive_y,
                         gif_path, fps=args.gif_fps, duration=args.duration)


if __name__ == '__main__':
    main()
