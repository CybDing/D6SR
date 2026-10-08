#!/usr/bin/env python3
"""
Evaluate trained RL policy on rough terrain (roughness=0.15).

Runs the same experiment as control/run_terrain_test.py but with the RL
residual policy on top of PID. Generates separate plots for:
  1. Angle tracking error (3-axis stability with mean/std bands)
  2. Pre-allocation control signals (3 attitude torques + 2 drive)

Usage:
    python rl/eval_terrain.py --roughness 0.15
    python rl/eval_terrain.py --roughness 0.15 --policy rl/checkpoints/best.pt
    python rl/eval_terrain.py --roughness 0.15 --drive-x 0.5 --drive-y 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import RobotSimulation
from rl.env import SphericalRobotEnv, OBS_DIM_FULL, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from control.trajectory import LineTrajectory, TrajectoryTracker
from terrain import RoughTerrain


def run_eval(env, agent, controller, duration, terrain):
    """Run terrain test with RL policy, return log dict."""
    # Rebuild simulation with terrain
    sim = RobotSimulation(
        mu_roll=env.mu_roll, force_max=env.force_max, terrain=terrain)
    sim.reset()
    # Replace env's sim
    env.sim = sim
    env.shell_id = sim._shell_body_id

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / env.pid_rate / dt)))
    total = int(duration / dt)
    shell_id = env.shell_id

    log = {'t': [], 'rpy': [], 'pos': [], 'pre_alloc': [], 'ctrl': [],
           'residual': [], 'joint_angles': []}
    ctrl_total = np.zeros(6)
    residual = np.zeros(env.act_dim)

    # Settle (short to minimize drift on rough terrain)
    settle_steps = int(0.5 / dt)
    for _ in range(settle_steps):
        sim.step()

    controller.reset()
    controller.set_target(roll=0, pitch=0, yaw=0)  # target yaw=0 (world frame)
    env.target_yaw = 0.0  # sync for RL obs
    controller.enabled = True

    t_start = sim.time  # zero-reference after settle

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            imu = sim.get_imu()

            # PID step
            pid_ctrl = controller.update(imu, ball_vel)

            # RL observation and action
            obs = env._get_obs()
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

            # Apply residual
            residual = env.alpha * action

            if env.unidirectional and env.act_dim == 3:
                # Combine PID attitude + RL residual at pre-alloc level
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

            sim.set_ctrl(ctrl_total)

        if i % 20 == 0:
            imu = sim.get_imu()
            log['t'].append(sim.time - t_start)
            log['rpy'].append(imu['euler'].copy())
            log['pos'].append(sim.data.xpos[shell_id].copy())
            log['pre_alloc'].append(controller.last_pre_alloc.copy())
            log['ctrl'].append(ctrl_total.copy())
            log['residual'].append(residual.copy())
            j1, _ = sim.get_joint_state('joint1')
            j2, _ = sim.get_joint_state('joint2')
            j3, _ = sim.get_joint_state('joint3')
            log['joint_angles'].append(np.array([j1, j2, j3]))

        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            pos = sim.data.xpos[shell_id]
            print(f"\rt={sim.time - t_start:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_eval_traj(env, agent, controller, duration, terrain, tracker,
                  traj_start_time, start_offset=(0.0, 0.0)):
    """Run terrain test with RL policy + trajectory tracking, return log dict."""
    sim = RobotSimulation(
        mu_roll=env.mu_roll, force_max=env.force_max, terrain=terrain)
    sim.reset()

    # Inject XY starting offset (freejoint qpos[0:2] = XY)
    if start_offset[0] != 0.0 or start_offset[1] != 0.0:
        sim.data.qpos[0] = float(start_offset[0])
        sim.data.qpos[1] = float(start_offset[1])
        mujoco.mj_forward(sim.model, sim.data)

    env.sim = sim
    env.shell_id = sim._shell_body_id

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / env.pid_rate / dt)))
    total = int(duration / dt)
    shell_id = env.shell_id

    log = {'t': [], 'rpy': [], 'pos': [], 'pre_alloc': [], 'ctrl': [],
           'ref': [], 'error': [], 'residual': [], 'joint_angles': []}
    ctrl_total = np.zeros(6)
    residual = np.zeros(env.act_dim)

    # Settle (short — minimize drift on rough terrain)
    settle_steps = int(0.5 / dt)
    for _ in range(settle_steps):
        sim.step()

    # Reset all PID states to avoid leftover integral from previous runs
    controller.reset()
    tracker.reset()

    controller.set_target(roll=0, pitch=0, yaw=0)  # target yaw=0 (world frame)
    env.target_yaw = 0.0  # sync for RL obs
    controller.enabled = True

    # Pre-stabilize: run controller + tracker at origin for 1s before logging
    pre_stab_steps = int(1.0 / dt)
    for i in range(pre_stab_steps):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            imu = sim.get_imu()
            vx, vy, _, _ = tracker.compute(0, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)
            pid_ctrl = controller.update(imu, ball_vel)
            obs = env._get_obs()
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            residual = env.alpha * action
            if env.unidirectional and env.act_dim == 3:
                pre = controller.last_pre_alloc
                ctrl_total = np.clip(
                    mix_motors(pre[0] + residual[0], pre[1] + residual[1],
                               pre[2] + residual[2], pre[3], pre[4],
                               unidirectional=True), -1.0, 1.0)
            else:
                residual_6 = env._mix_residual(residual)
                ctrl_total = np.clip(pid_ctrl + residual_6, -1.0, 1.0)
            sim.set_ctrl(ctrl_total)
        sim.step()

    # Reset tracker integral after pre-stabilization so trajectory starts clean
    tracker.reset()
    t_start = sim.time

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            imu = sim.get_imu()
            traj_t = sim.time - t_start

            # Trajectory tracker → velocity commands
            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            # PID step
            pid_ctrl = controller.update(imu, ball_vel)

            # RL observation and action
            obs = env._get_obs()
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

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

            sim.set_ctrl(ctrl_total)

        if i % 20 == 0:
            imu = sim.get_imu()
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['rpy'].append(imu['euler'].copy())
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)
            log['pre_alloc'].append(controller.last_pre_alloc.copy())
            log['ctrl'].append(ctrl_total.copy())
            log['residual'].append(residual.copy())
            j1, _ = sim.get_joint_state('joint1')
            j2, _ = sim.get_joint_state('joint2')
            j3, _ = sim.get_joint_state('joint3')
            log['joint_angles'].append(np.array([j1, j2, j3]))

        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            ball_pos = sim.data.xpos[shell_id][:2]
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_baseline_traj(controller, duration, terrain, tracker, mu_roll, force_max,
                      start_offset=(0.0, 0.0)):
    """Run trajectory tracking with drive-only (no attitude PID).

    controller.enabled is set to False so attitude outputs = 0, but velocity
    PIDs still produce drive commands.  No RL agent needed.
    """
    sim = RobotSimulation(mu_roll=mu_roll, force_max=force_max, terrain=terrain)
    sim.reset()

    # Inject XY starting offset
    if start_offset[0] != 0.0 or start_offset[1] != 0.0:
        sim.data.qpos[0] = float(start_offset[0])
        sim.data.qpos[1] = float(start_offset[1])
        mujoco.mj_forward(sim.model, sim.data)

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / controller.pid_rate / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id

    log = {'t': [], 'rpy': [], 'pos': [], 'ref': [], 'error': []}
    ctrl_total = np.zeros(6)

    # Settle
    settle_steps = int(0.5 / dt)
    for _ in range(settle_steps):
        sim.step()

    controller.reset()
    tracker.reset()
    controller.set_target(roll=0, pitch=0, yaw=0)
    controller.enabled = False  # no attitude stabilization

    t_start = sim.time

    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            imu = sim.get_imu()
            traj_t = sim.time - t_start

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            pid_ctrl = controller.update(imu, ball_vel)
            ctrl_total = np.clip(pid_ctrl, -1.0, 1.0)
            sim.set_ctrl(ctrl_total)

        if i % 20 == 0:
            imu = sim.get_imu()
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['rpy'].append(imu['euler'].copy())
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)

        if i % int(0.5 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            ball_pos = sim.data.xpos[shell_id][:2]
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                  f"err={err:.3f}m  pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    controller.enabled = True  # restore
    return {k: np.array(v) for k, v in log.items()}


def run_viewer_eval(env, agent, controller, terrain, tracker=None):
    """Interactive viewer with RL policy (+ optional trajectory tracking)."""
    import mujoco.viewer

    sim = RobotSimulation(
        mu_roll=env.mu_roll, force_max=env.force_max, terrain=terrain)
    sim.reset()
    env.sim = sim
    env.shell_id = sim._shell_body_id

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / env.pid_rate / dt)))
    shell_id = env.shell_id
    ctrl_total = np.zeros(6)

    # Settle (short to minimize drift on rough terrain)
    settle_steps = int(0.5 / dt)
    for _ in range(settle_steps):
        sim.step()

    controller.reset()
    if tracker is not None:
        tracker.reset()
    controller.set_target(roll=0, pitch=0, yaw=0)  # target yaw=0 (world frame)
    env.target_yaw = 0.0  # sync for RL obs
    controller.enabled = True
    t_start = sim.time
    step_i = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            t0 = time.perf_counter()

            if step_i % steps_per_pid == 0:
                ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
                imu = sim.get_imu()

                if tracker is not None:
                    ball_pos = sim.data.xpos[shell_id][:2].copy()
                    traj_t = sim.time - t_start
                    vx, vy, ref_x, ref_y = tracker.compute(
                        traj_t, ball_pos[0], ball_pos[1])
                    controller.set_drive(vx, vy)

                pid_ctrl = controller.update(imu, ball_vel)

                obs = env._get_obs()
                action = agent.select_action(obs, deterministic=True)
                action = np.clip(np.asarray(action, dtype=np.float64),
                                 -1.0, 1.0)
                residual = env.alpha * action

                if env.unidirectional and env.act_dim == 3:
                    pre = controller.last_pre_alloc
                    ctrl_total = np.clip(
                        mix_motors(pre[0] + residual[0],
                                   pre[1] + residual[1],
                                   pre[2] + residual[2],
                                   pre[3], pre[4], unidirectional=True),
                        -1.0, 1.0)
                else:
                    residual_6 = env._mix_residual(residual)
                    ctrl_total = np.clip(pid_ctrl + residual_6, -1.0, 1.0)

                sim.set_ctrl(ctrl_total)

            if step_i % int(0.5 / dt) == 0:
                r, p, y = sim.get_imu()['euler']
                pos = sim.data.xpos[shell_id]
                if tracker is not None:
                    traj_t = sim.time - t_start
                    rx, ry = tracker.traj.position(traj_t)
                    err = np.hypot(pos[0] - rx, pos[1] - ry)
                    print(f"\rt={traj_t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                          f"err={err:.3f}m  pos=({pos[0]:+.2f},{pos[1]:+.2f})",
                          end="", flush=True)
                else:
                    print(f"\rt={sim.time - t_start:5.1f}s  R={r:+6.1f} P={p:+6.1f} "
                          f"Y={y:+6.1f}  pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})",
                          end="", flush=True)

            sim.step()
            viewer.sync()
            step_i += 1
            sl = dt - (time.perf_counter() - t0)
            if sl > 0:
                time.sleep(sl)
    print()


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
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
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
    pre_alloc = log['pre_alloc']

    for i in range(3):
        rpy[:, i] = np.degrees(np.unwrap(np.radians(rpy[:, i])))

    print(f"Saving RL terrain plots to {plot_dir}/")

    # ---- Fig 1: Angle Tracking Error (stability) ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'RL Angle Stability: {terrain_name} terrain', fontsize=13)
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
    _save_fig(fig, f'rl_terrain_{terrain_name}_angle_stability', plot_dir)

    # ---- Fig 2: Pre-allocation Control Signals (PID + RL) ----
    res = log.get('residual')
    has_rl = res is not None and res.shape[0] == pre_alloc.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f'RL Pre-allocation Control Signals (PID + RL Residual): '
        f'{terrain_name} terrain', fontsize=13)

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
    _save_fig(fig, f'rl_terrain_{terrain_name}_pre_alloc_signals', plot_dir)

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
    ax.set_title(f'RL Motor Ctrl Signals [{ids_str}]: {terrain_name} terrain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, f'rl_terrain_{terrain_name}_motors_{ids_str}', plot_dir)
    _safe_show(plt)


def plot_terrain_trajectory(log_or_logs, terrain, terrain_name):
    """Unified trajectory figure: terrain+paths (left), height profile +
    error time series (right).  Accepts a single log dict or a list of logs
    (multi-start).  Sized for IEEE two-column (7.16 in width)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available")
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })

    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)

    # Normalize to list of logs
    if isinstance(log_or_logs, dict):
        logs = [log_or_logs]
    else:
        logs = log_or_logs
    n_runs = len(logs)
    multi = n_runs > 1

    # ── Bounding box across all trajectories ──
    all_pos_x, all_traj_y = [], []
    for log in logs:
        all_pos_x.append(log['pos'][:, 0])
        all_pos_x.append(log['ref'][:, 0])
        all_traj_y.append(log['pos'][:, 1])
    all_x = np.concatenate(all_pos_x)
    all_y = np.concatenate(all_traj_y)

    ref_y_center = np.mean(logs[0]['ref'][:, 1])
    max_lateral_err = max(np.max(log['error']) for log in logs)
    y_span = max(np.ptp(all_y), max_lateral_err * 3.0, 1.0)

    pad_x = max(2.0, 0.10 * np.ptp(all_x))
    x_min, x_max = all_x.min() - pad_x, all_x.max() + pad_x
    y_half = y_span / 2 + 0.3
    y_min, y_max = ref_y_center - y_half, ref_y_center + y_half

    # ── Terrain heightmap ──
    heights_m = terrain.generate_heights() * terrain.elevation_max
    cr, cc = terrain.nrow // 2, terrain.ncol // 2
    heights_m -= heights_m[cr, cc]
    x_coords = np.linspace(-terrain.size_x, terrain.size_x, terrain.ncol)
    y_coords = np.linspace(-terrain.size_y, terrain.size_y, terrain.nrow)

    xi_min = max(0, np.searchsorted(x_coords, x_min) - 1)
    xi_max = min(len(x_coords), np.searchsorted(x_coords, x_max) + 1)
    yi_min = max(0, np.searchsorted(y_coords, y_min) - 1)
    yi_max = min(len(y_coords), np.searchsorted(y_coords, y_max) + 1)
    x_crop = x_coords[xi_min:xi_max]
    y_crop = y_coords[yi_min:yi_max]
    h_crop = heights_m[yi_min:yi_max, xi_min:xi_max]

    # ── Height profile along reference X ──
    ref = logs[0]['ref']
    ref_x_vals = ref[:, 0]
    # Sample terrain height at (ref_x, ref_y_center)
    h_along_ref = np.interp(ref_x_vals, x_coords,
                            heights_m[terrain.nrow // 2, :])

    print(f"Saving RL terrain trajectory plot to {plot_dir}/")

    # ── Figure layout: IEEE two-column = 7.16in wide ──
    fig_w = 7.16
    fig_h = 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3, 1],
                           height_ratios=[1, 1],
                           wspace=0.30, hspace=0.45)

    ax_traj = fig.add_subplot(gs[:, 0])    # left: full-height trajectory
    ax_hgt  = fig.add_subplot(gs[0, 1])    # top-right: height profile
    ax_err  = fig.add_subplot(gs[1, 1])    # bottom-right: error time series

    # ════════════════════════════════════════════════════
    # Left panel: terrain heightmap + trajectory paths
    # ════════════════════════════════════════════════════
    Xg, Yg = np.meshgrid(x_crop, y_crop)
    pcm = ax_traj.pcolormesh(Xg, Yg, h_crop, cmap='YlGnBu', alpha=0.45,
                              shading='auto')

    if np.ptp(h_crop) > 1e-6:
        n_levels = min(8, max(3, int(np.ptp(h_crop) / 0.005)))
        ax_traj.contour(Xg, Yg, h_crop, levels=n_levels, colors='k',
                        linewidths=0.25, alpha=0.25)

    # Reference line
    ax_traj.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.5,
                 label='Reference', zorder=5)

    if multi:
        # Multi-start: each trajectory as a distinct color
        cmap_tab10 = plt.cm.tab10
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            pos = log['pos']
            lbl = f'Run {idx + 1}'
            ax_traj.plot(pos[:, 0], pos[:, 1], color=clr, linewidth=1.0,
                         alpha=0.85, label=lbl, zorder=6)
            ax_traj.plot(pos[0, 0], pos[0, 1], 'o', color=clr, markersize=4,
                         markeredgecolor='k', markeredgewidth=0.6, zorder=7)
            ax_traj.plot(pos[-1, 0], pos[-1, 1], 's', color=clr, markersize=3.5,
                         markeredgecolor='k', markeredgewidth=0.6, zorder=7)
    else:
        # Single run: color-coded by error
        log = logs[0]
        pos, error = log['pos'], log['error']
        err_norm = Normalize(vmin=0, vmax=max(error.max(), 1e-3))
        err_cmap = plt.cm.coolwarm
        for j in range(len(pos) - 1):
            ax_traj.plot(pos[j:j+2, 0], pos[j:j+2, 1],
                         color=err_cmap(err_norm(error[j])),
                         linewidth=1.5, zorder=6)
        sm = ScalarMappable(norm=err_norm, cmap=err_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_traj, label='Lateral Error (m)',
                     shrink=0.4, pad=0.02, aspect=20)
        ax_traj.plot(pos[0, 0], pos[0, 1], 'o', color='lime', markersize=7,
                     markeredgecolor='k', markeredgewidth=0.8,
                     label='Start', zorder=7)
        ax_traj.plot(pos[-1, 0], pos[-1, 1], 's', color='orange', markersize=7,
                     markeredgecolor='k', markeredgewidth=0.8,
                     label='End', zorder=7)

    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    if multi:
        ax_traj.set_title(f'Trajectory on {terrain_name} terrain '
                          f'({n_runs} starts)')
    else:
        ax_traj.set_title(f'Trajectory on {terrain_name} terrain')
    ax_traj.legend(loc='upper left', fontsize=6,
                   ncol=min(n_runs + 1, 3), handlelength=1.5)
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_aspect('auto')

    # ════════════════════════════════════════════════════
    # Top-right: terrain height profile along path
    # ════════════════════════════════════════════════════
    ax_hgt.fill_between(ref_x_vals, 0, h_along_ref * 1000,
                        color='#8DB4C2', alpha=0.4)
    ax_hgt.plot(ref_x_vals, h_along_ref * 1000, color='#2E6B7E',
                linewidth=0.8)
    ax_hgt.set_ylabel('Height (mm)')
    ax_hgt.set_title('Terrain profile along path')
    ax_hgt.grid(True, alpha=0.25, linewidth=0.4)
    ax_hgt.set_xlim(ref_x_vals.min(), ref_x_vals.max())

    # ════════════════════════════════════════════════════
    # Bottom-right: tracking error time series
    # ════════════════════════════════════════════════════
    if multi:
        cmap_tab10 = plt.cm.tab10
        max_t = 0
        all_t_arrays, all_err_arrays = [], []
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            ax_err.plot(log['t'], log['error'], color=clr, alpha=0.4,
                        linewidth=0.6)
            all_t_arrays.append(log['t'])
            all_err_arrays.append(log['error'])
            max_t = max(max_t, log['t'][-1])

        t_common = np.linspace(0, max_t, 500)
        interp_errors = np.array([
            np.interp(t_common, ta, ea)
            for ta, ea in zip(all_t_arrays, all_err_arrays)])
        mean_err = np.mean(interp_errors, axis=0)
        std_err = np.std(interp_errors, axis=0)
        ax_err.fill_between(t_common, np.maximum(mean_err - std_err, 0),
                            mean_err + std_err, color='gray', alpha=0.25)
        ax_err.plot(t_common, mean_err, 'k-', linewidth=1.0, alpha=0.7,
                    label=f'Mean={np.mean(mean_err):.3f}m')
    else:
        log = logs[0]
        t, error = log['t'], log['error']
        ax_err.plot(t, error, color='tab:blue', alpha=0.8, linewidth=0.8)
        mean_err_val = np.mean(error)
        ax_err.axhline(mean_err_val, color='tab:red', ls='--', alpha=0.5,
                       linewidth=0.7,
                       label=f'Mean={mean_err_val:.3f}m')
        ax_err.fill_between(t,
                            max(mean_err_val - np.std(error), 0),
                            mean_err_val + np.std(error),
                            color='tab:blue', alpha=0.08)

    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error (m)')
    ax_err.set_title('Tracking error')
    ax_err.legend(fontsize=6)
    ax_err.grid(True, alpha=0.25, linewidth=0.4)

    # ── Save ──
    suffix = f'_multistart_{n_runs}runs' if multi else ''
    _save_fig(fig, f'rl_terrain_{terrain_name}_trajectory{suffix}', plot_dir)
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
    # joint2 axis in Link1 frame: [-0.84332, 0, 0.53742]
    # Lost axis in base frame (≈world): [sin(θ₁-φ), 0, -cos(θ₁-φ)]
    phi = np.arctan2(0.53742, 0.84332)  # ≈ 32.5° tilt offset

    lock_severity = 1.0 - np.abs(np.sin(ja[:, 1]))  # 1=locked, 0=free
    roll_lock_frac = np.abs(np.sin(ja[:, 0] - phi))  # X-projection
    yaw_lock_frac  = np.abs(np.cos(ja[:, 0] - phi))  # Z-projection

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

    # Background: red = Roll locked, blue = Yaw locked
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

    # Overlay per-axis |RL residual|
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


def plot_comparison_figure(baseline_logs, rl_logs, terrain, terrain_name):
    """Comparison figure: no attitude control vs PID+RL (multi-start).

    Layout — two independent GridSpecs:
      gs_top (2×2, width_ratios=[2.5,1]):
        [0,0] — (a) No-control trajectories
        [1,0] — (b) PID+RL trajectories
        [0:2,1] — 2D terrain heightmap + colorbar
      gs_bot (1×2, equal columns, full figure width):
        [0,0] — Tracking error no-control: mean±std
        [0,1] — Tracking error PID+RL: mean±std

    Each trajectory panel uses its own Y range = y_span * 1.5.
    Bottom error panels are equal-width and span the full figure width.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available")
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })

    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    n_runs = len(baseline_logs)
    ref = baseline_logs[0]['ref']
    ref_x_vals = ref[:, 0]
    ref_y_center = float(np.mean(ref[:, 1]))

    # ── Per-panel axis bounds (independent Y range = y_span * 1.5) ──
    def _panel_xy_bounds(logs):
        pos_x = np.concatenate([log['pos'][:, 0] for log in logs])
        pos_y = np.concatenate([log['pos'][:, 1] for log in logs])
        ref_x = ref[:, 0]
        y_span = max(np.ptp(pos_y), 1.0)
        y_half = y_span * 1.5 / 2
        pad_x = max(2.0, 0.10 * np.ptp(np.concatenate([pos_x, ref_x])))
        x_lo = min(pos_x.min(), ref_x.min()) - pad_x
        x_hi = max(pos_x.max(), ref_x.max()) + pad_x
        return x_lo, x_hi, ref_y_center - y_half, ref_y_center + y_half

    bl_x0, bl_x1, bl_y0, bl_y1 = _panel_xy_bounds(baseline_logs)
    rl_x0, rl_x1, rl_y0, rl_y1 = _panel_xy_bounds(rl_logs)

    # ── Terrain heightmap (crop to union of both trajectory panels) ──
    heights_m = terrain.generate_heights() * terrain.elevation_max
    cr, cc = terrain.nrow // 2, terrain.ncol // 2
    heights_m -= heights_m[cr, cc]
    x_coords = np.linspace(-terrain.size_x, terrain.size_x, terrain.ncol)
    y_coords = np.linspace(-terrain.size_y, terrain.size_y, terrain.nrow)

    xc_lo = min(bl_x0, rl_x0)
    xc_hi = max(bl_x1, rl_x1)
    yc_lo = min(bl_y0, rl_y0)
    yc_hi = max(bl_y1, rl_y1)
    xi_min = max(0, np.searchsorted(x_coords, xc_lo) - 1)
    xi_max = min(len(x_coords), np.searchsorted(x_coords, xc_hi) + 1)
    yi_min = max(0, np.searchsorted(y_coords, yc_lo) - 1)
    yi_max = min(len(y_coords), np.searchsorted(y_coords, yc_hi) + 1)
    x_crop = x_coords[xi_min:xi_max]
    y_crop = y_coords[yi_min:yi_max]
    h_crop = heights_m[yi_min:yi_max, xi_min:xi_max]
    Xg, Yg = np.meshgrid(x_crop, y_crop)

    # Height range in mm — shared across both trajectory panels for consistent color
    h_mm = h_crop * 1000
    h_vmin, h_vmax = float(h_mm.min()), float(h_mm.max())

    print(f"Saving comparison figure to {plot_dir}/")

    # ── Figure: two independent GridSpecs for top (traj) and bottom (errors) ──
    fig = plt.figure(figsize=(7.16, 5.5))

    # Top portion: 2 stacked trajectory panels.
    # right=0.87 leaves a strip on the right for the shared terrain colorbar.
    gs_top = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.07, right=0.87,
        bottom=0.40, top=0.97,
        hspace=0.52)

    # Colorbar axes: narrow vertical strip, aligned with both traj rows.
    cbar_ax = fig.add_axes([0.895, 0.40, 0.022, 0.57])

    # Bottom portion: 2 equal-width error panels spanning the same left–right range.
    gs_bot = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.07, right=0.96,
        bottom=0.06, top=0.29,
        wspace=0.32)

    ax_bl  = fig.add_subplot(gs_top[0, 0])  # (a) no-control trajectories
    ax_rl  = fig.add_subplot(gs_top[1, 0])  # (b) PID+RL agent trajectories
    ax_ebl = fig.add_subplot(gs_bot[0, 0])  # tracking error no-control
    ax_erl = fig.add_subplot(gs_bot[0, 1])  # tracking error PID+RL agent

    cmap_tab10 = plt.cm.tab10

    # ── Trajectory panels ──
    def _draw_traj_panel(ax, logs, title_label, x0, x1, y0, y1):
        ax.pcolormesh(Xg, Yg, h_mm, cmap='YlGnBu', alpha=0.45,
                      shading='auto', vmin=h_vmin, vmax=h_vmax)
        if np.ptp(h_mm) > 1e-3:
            n_levels = min(8, max(3, int(np.ptp(h_mm) / 5)))
            ax.contour(Xg, Yg, h_mm, levels=n_levels, colors='k',
                       linewidths=0.25, alpha=0.25)
        ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.2,
                label='Ref.', zorder=5)
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            pos = log['pos']
            ax.plot(pos[:, 0], pos[:, 1], color=clr, linewidth=1.0,
                    alpha=0.85, label=f'Run {idx + 1}', zorder=6)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=clr, markersize=3,
                    markeredgecolor='k', markeredgewidth=0.5, zorder=7)
            ax.plot(pos[-1, 0], pos[-1, 1], 's', color=clr, markersize=2.5,
                    markeredgecolor='k', markeredgewidth=0.5, zorder=7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('auto')
        ax.set_ylabel('Y (m)')
        ax.set_title(title_label)
        ax.legend(loc='upper left', fontsize=5, ncol=n_runs + 1,
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
                  borderpad=0.3, framealpha=0.85)

    _draw_traj_panel(ax_bl, baseline_logs, '(a) No attitude control',
                     bl_x0, bl_x1, bl_y0, bl_y1)
    _draw_traj_panel(ax_rl, rl_logs, '(b) PID + RL agent',
                     rl_x0, rl_x1, rl_y0, rl_y1)
    ax_rl.set_xlabel('X (m)')

    # ── Shared terrain colorbar (right strip, spans both traj rows) ──
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    norm = Normalize(vmin=h_vmin, vmax=h_vmax)
    sm = ScalarMappable(cmap='YlGnBu', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Terrain height (mm)', fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5)

    # ── Error panels (equal width, spanning full figure width) ──
    def _draw_error_panel(ax, logs, title_label):
        all_t_arrays, all_err_arrays = [], []
        max_t = 0
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            ax.plot(log['t'], log['error'], color=clr, alpha=0.35,
                    linewidth=0.5)
            all_t_arrays.append(log['t'])
            all_err_arrays.append(log['error'])
            max_t = max(max_t, log['t'][-1])
        t_common = np.linspace(0, max_t, 500)
        interp_errors = np.array([
            np.interp(t_common, ta, ea)
            for ta, ea in zip(all_t_arrays, all_err_arrays)])
        mean_err = np.mean(interp_errors, axis=0)
        std_err  = np.std(interp_errors, axis=0)
        overall_mean = float(np.mean(mean_err))
        ax.fill_between(t_common, np.maximum(mean_err - std_err, 0),
                        mean_err + std_err, color='gray', alpha=0.25)
        ax.plot(t_common, mean_err, 'k-', linewidth=1.0, alpha=0.8,
                label=f'Mean = {overall_mean:.3f} m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title(title_label)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.25, linewidth=0.4)

    _draw_error_panel(ax_ebl, baseline_logs, '(c) Error — No attitude control')
    _draw_error_panel(ax_erl, rl_logs,       '(d) Error — PID + RL agent')

    _save_fig(fig, f'rl_terrain_{terrain_name}_comparison_{n_runs}runs',
              plot_dir)
    _safe_show(plt)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL policy on rough terrain")
    parser.add_argument('--roughness', type=float, default=0.15,
                        help='Rough terrain roughness (m)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hfield-res', type=int, default=600)
    parser.add_argument('--hfield-size', type=float, default=100.0)
    parser.add_argument('--drive-x', type=float, default=0.5)
    parser.add_argument('--drive-y', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=50.0)
    parser.add_argument('--pid-rate', type=float, default=50.0)
    parser.add_argument('--mu-roll', type=float, default=0.01)
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to trained policy checkpoint')
    parser.add_argument('--act-dim', type=int, default=3, choices=[3, 5])
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--unidirectional', action='store_true',
                        help='Unidirectional motor mixing (no reverse)')
    # Trajectory tracking
    parser.add_argument('--trajectory', action='store_true',
                        help='Enable line trajectory tracking mode')
    parser.add_argument('--speed', type=float, default=0.5,
                        help='Travel speed along line (m/s)')
    parser.add_argument('--line-length', type=float, default=30.0,
                        help='Line trajectory length (m)')
    parser.add_argument('--line-angle', type=float, default=0.0,
                        help='Line direction in degrees from +X axis')
    # Multi-start robustness
    parser.add_argument('--multi-start', type=int, default=4,
                        help='Number of multi-start runs (0=disabled)')
    parser.add_argument('--multi-start-radius', type=float, default=0.5,
                        help='Y offset radius for multi-start (m)')
    parser.add_argument('--plot-motors', type=int, nargs='+', default=None,
                        metavar='ID',
                        help='Motor indices to plot (e.g. --plot-motors 0 1 4 5)')
    parser.add_argument('--compare', action='store_true',
                        help='Run no-attitude-control baseline alongside PID+RL')
    parser.add_argument('--headless', action='store_true',
                        help='Run headless (no viewer)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots (implies --headless)')
    parser.add_argument('--yaw-compensate', action='store_true',
                        help='Rotate drive commands to body frame using current yaw')
    args = parser.parse_args()
    if args.plot or args.plot_motors:
        args.headless = True

    if args.plot_motors:
        for mid in args.plot_motors:
            if mid < 0 or mid > 5:
                parser.error(f"Motor ID must be 0-5, got {mid}")

    if args.policy is None:
        # args.policy = str(Path(__file__).parent / 'checkpoints' / 'no_joint' / 'best.pt')
        args.policy = str(Path(__file__).parent / 'checkpoints' / 'old1' /'best.pt')
        # args.policy = str(Path(__file__).parent / 'checkpoints' /'best.pt')


    # Auto-set duration for trajectory mode
    if args.trajectory:
        auto_dur = args.line_length / max(args.speed, 1e-3) + 5.0
        if args.duration == 50.0:  # default
            args.duration = auto_dur

    # Load RL agent — auto-detect obs_dim from checkpoint
    ckpt_info = peek_checkpoint(args.policy)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
    agent.load(args.policy)
    print(f"Loaded RL policy from {args.policy} (act_dim={args.act_dim}, "
          f"obs_dim={obs_dim}{', no_joint' if no_joint else ''})")

    # Create rough terrain
    terrain = RoughTerrain(
        roughness=args.roughness, seed=args.seed,
        nrow=args.hfield_res, ncol=args.hfield_res,
        size_x=args.hfield_size, size_y=args.hfield_size,
    )
    terrain_name = f'rough_{args.roughness}'
    print(f"Terrain: rough (roughness={args.roughness})")

    # Create env (terrain will be injected later in run_eval)
    env = SphericalRobotEnv(
        mu_roll=args.mu_roll,
        force_max=5,
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

    controller = env.controller
    controller.yaw_compensate = args.yaw_compensate

    if args.trajectory:
        # Trajectory tracking mode
        angle_rad = np.radians(args.line_angle)
        end_x = args.line_length * np.cos(angle_rad)
        end_y = args.line_length * np.sin(angle_rad)
        traj = LineTrajectory(start=(0, 0), end=(end_x, end_y),
                              speed=args.speed)
        tracker = TrajectoryTracker(traj, dt=1.0/args.pid_rate)

        print(f"Running line trajectory (length={args.line_length}m, "
              f"angle={args.line_angle}°, speed={args.speed}m/s, "
              f"duration={args.duration:.1f}s)...")

        if args.headless:
            log = run_eval_traj(env, agent, controller, args.duration, terrain,
                                tracker, traj_start_time=0)

            # Tracking stats
            error = log['error']
            n = len(error) // 4
            print(f"\nRL Tracking stats (after transient):")
            print(f"  Error: mean={np.mean(error[n:]):.3f}m  "
                  f"std={np.std(error[n:]):.3f}m  "
                  f"max={np.max(error[n:]):.3f}m")
            rpy = log['rpy']
            for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
                arr = rpy[n:, i]
                print(f"  {name:6s}: mean={np.mean(arr):+.2f}  "
                      f"std={np.std(arr):.2f}  max={np.max(np.abs(arr)):.2f}")

            if args.plot:
                plot_log(log, terrain_name)
                plot_terrain_trajectory(log, terrain, terrain_name)
                plot_residual_analysis(log, terrain_name)
            if args.plot_motors:
                plot_motors(log, terrain_name, args.plot_motors)

            # Multi-start robustness evaluation
            if args.multi_start > 0 and args.plot:
                n_ms = args.multi_start
                radius = args.multi_start_radius
                offsets = np.linspace(-radius, radius, n_ms)

                if args.compare:
                    # ── Comparison mode: baseline (no attitude) vs PID+RL ──
                    baseline_logs = []
                    rl_logs = []
                    print(f"\nComparison multi-start: {n_ms} runs, Y offsets in "
                          f"[{-radius:+.2f}, {+radius:+.2f}]m")
                    for oi, y_off in enumerate(offsets):
                        print(f"  Run {oi+1}/{n_ms}: y_offset={y_off:+.3f}m")

                        # --- Baseline (no attitude control) ---
                        print(f"    [baseline] no attitude control...")
                        bl_traj = LineTrajectory(
                            start=(0, 0), end=(end_x, end_y),
                            speed=args.speed)
                        bl_tracker = TrajectoryTracker(
                            bl_traj, dt=1.0/args.pid_rate)
                        controller.reset()
                        bl_log = run_baseline_traj(
                            controller, args.duration, terrain, bl_tracker,
                            mu_roll=args.mu_roll, force_max=5,
                            start_offset=(0.0, y_off))
                        baseline_logs.append(bl_log)
                        err = bl_log['error']
                        n_q = len(err) // 4
                        print(f"    baseline err: mean={np.mean(err[n_q:]):.3f}m  "
                              f"max={np.max(err[n_q:]):.3f}m")

                        # --- PID + RL ---
                        print(f"    [PID+RL] attitude stabilization...")
                        rl_traj = LineTrajectory(
                            start=(0, 0), end=(end_x, end_y),
                            speed=args.speed)
                        rl_tracker = TrajectoryTracker(
                            rl_traj, dt=1.0/args.pid_rate)
                        controller.reset()
                        rl_log = run_eval_traj(
                            env, agent, controller, args.duration, terrain,
                            rl_tracker, traj_start_time=0,
                            start_offset=(0.0, y_off))
                        rl_logs.append(rl_log)
                        err = rl_log['error']
                        n_q = len(err) // 4
                        print(f"    PID+RL err: mean={np.mean(err[n_q:]):.3f}m  "
                              f"max={np.max(err[n_q:]):.3f}m")

                    plot_comparison_figure(
                        baseline_logs, rl_logs, terrain, terrain_name)
                else:
                    # ── Standard multi-start (PID+RL only) ──
                    multi_logs = []
                    print(f"\nMulti-start: {n_ms} runs, Y offsets in "
                          f"[{-radius:+.2f}, {+radius:+.2f}]m")
                    for oi, y_off in enumerate(offsets):
                        print(f"  Run {oi+1}/{n_ms}: y_offset={y_off:+.3f}m")
                        ms_traj = LineTrajectory(
                            start=(0, 0), end=(end_x, end_y),
                            speed=args.speed)
                        ms_tracker = TrajectoryTracker(
                            ms_traj, dt=1.0/args.pid_rate)
                        controller.reset()
                        ms_log = run_eval_traj(
                            env, agent, controller, args.duration, terrain,
                            ms_tracker, traj_start_time=0,
                            start_offset=(0.0, y_off))
                        multi_logs.append(ms_log)
                        err = ms_log['error']
                        n_q = len(err) // 4
                        print(f"    Error: mean={np.mean(err[n_q:]):.3f}m  "
                              f"max={np.max(err[n_q:]):.3f}m")
                    plot_terrain_trajectory(
                        multi_logs, terrain, terrain_name)
        else:
            run_viewer_eval(env, agent, controller, terrain, tracker=tracker)
    else:
        # Original constant-velocity mode
        controller.set_drive(vx=args.drive_x, vy=args.drive_y)

        if args.headless:
            log = run_eval(env, agent, controller, args.duration, terrain)

            # Summary stats
            rpy = log['rpy']
            n = len(rpy) // 2
            print(f"\nRL Stability stats (second half):")
            for i, name in enumerate(['Roll', 'Pitch', 'Yaw']):
                arr = rpy[n:, i]
                print(f"  {name:6s}: mean={np.mean(arr):+.2f}  "
                      f"std={np.std(arr):.2f}  max={np.max(np.abs(arr)):.2f}")
            pos = log['pos']
            print(f"  Drift:  dX={pos[-1, 0] - pos[0, 0]:+.3f}  "
                  f"dY={pos[-1, 1] - pos[0, 1]:+.3f}  "
                  f"dZ={pos[-1, 2] - pos[0, 2]:+.3f} m")

            if args.plot:
                plot_log(log, terrain_name)
                plot_residual_analysis(log, terrain_name)
            if args.plot_motors:
                plot_motors(log, terrain_name, args.plot_motors)
        else:
            run_viewer_eval(env, agent, controller, terrain)


if __name__ == "__main__":
    main()
