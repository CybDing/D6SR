#!/usr/bin/env python3
"""
Train and evaluate residual RL (SAC) on top of PID baseline.

Training: flat ground, random velocity targets, SAC learns residual corrections.
Evaluation: compare PID-only vs PID+residual on velocity tracking and trajectory.

Action modes:
    --act-dim 3  (default): attitude-only residual [roll, pitch, yaw]
    --act-dim 5:            full residual [roll, pitch, yaw, x_drive, y_drive]

Usage:
    python rl/train_residual.py                                    # train 300k steps (3-dim)
    python rl/train_residual.py --act-dim 5                        # train with drive residual
    python rl/train_residual.py --eval --plot --policy rl/checkpoints/best.pt
    mjpython rl/train_residual.py --viewer --policy rl/checkpoints/best.pt
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.env import SphericalRobotEnv, OBS_DIM_FULL, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint

CKPT_DIR = Path(__file__).parent / 'checkpoints'


# ======================================================================
# Training
# ======================================================================

def train(args):
    if getattr(args, 'ckpt_subdir', None):
        ckpt_dir = CKPT_DIR / args.ckpt_subdir
    else:
        ckpt_dir = CKPT_DIR / 'no_joint' if args.no_joint_obs else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    act_dim = args.act_dim
    no_joint = args.no_joint_obs
    env = SphericalRobotEnv(
        frame_stack=getattr(args, 'frame_stack', 1),
        alpha_max=args.alpha_max,
        warmup_steps=args.warmup,
        max_episode_steps=args.episode_steps,
        randomize=True,
        act_dim=act_dim,
        unidirectional=args.unidirectional,
        terrain_prob=args.terrain_prob,
        roughness_range=(args.roughness_min, args.roughness_max),
        terrain_switch_interval=args.terrain_switch_interval,
        no_joint_obs=no_joint,
        shell_inertia=getattr(args, 'shell_inertia', None),
        inertia_scales=getattr(args, '_inertia_scales', None),
        attitude_gain_scale=getattr(args, 'gain_scale', 1.0),
        attitude_gains=getattr(args, '_attitude_gains', None),
        mass_overrides=(getattr(args, '_sim_config', None) or {}).get('mass_overrides'),
        inertia_overrides=(getattr(args, '_sim_config', None) or {}).get('inertia_overrides'),
        pos_overrides=(getattr(args, '_sim_config', None) or {}).get('pos_overrides'),
        dof_damping_scale=(getattr(args, '_sim_config', None) or {}).get('dof_damping_scale'),
        joint_frictionloss=(getattr(args, '_sim_config', None) or {}).get('joint_frictionloss'),
        joint_damping=(getattr(args, '_sim_config', None) or {}).get('joint_damping'),
        force_max=(getattr(args, '_sim_config', None) or {}).get('force_max', 5),
        yaw_reward_weight=getattr(args, 'yaw_reward', 0.0),
        act_penalty=getattr(args, 'act_penalty', 10.0),
        smooth_penalty=getattr(args, 'smooth_penalty', 2.0),
        **(dict(imu_delay_steps=1, imu_angle_noise=0.5,
                imu_gyro_noise=0.02, thrust_tau=0.08)
           if getattr(args, 'realism', False) else {}),
    )
    obs_dim = env.obs_dim
    agent = SAC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=(128, 128),
        lr_actor=getattr(args, 'lr_actor', 3e-4),
        lr_critic=getattr(args, 'lr_critic', 1e-4),
        lr_alpha=3e-4,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        grad_clip=1.0,
        utd_ratio=args.utd_ratio,
        target_entropy=getattr(args, 'target_entropy', None),
        init_alpha=getattr(args, 'init_alpha', 1.0),
    )

    if args.resume:
        agent.load(args.resume)
        env.global_step = agent.train_step
        print(f"Resumed from {args.resume} (step {agent.train_step})")

    obs_label = f"obs_dim={obs_dim}" + (" (no joint angles)" if no_joint else " (with joint angles)")
    print(f"SAC config: act_dim={act_dim}  {obs_label}  batch={args.batch_size}  utd={args.utd_ratio}  "
          f"lr_actor=3e-4  lr_critic=1e-4  hidden=(128,128)  grad_clip=1.0")
    print(f"Env config: alpha_max={args.alpha_max}  warmup={args.warmup}  "
          f"ep_len={args.episode_steps}  randomize=True")
    print(f"Warmup: {args.warmup_random} steps zero-action (pure PID baseline)")

    # --fixed-drive: train on the eval task (fixed target velocity) instead of
    # randomized targets — cuts return variance so the critic can rank the
    # small residual advantages; terrain randomization stays on.
    reset_kw = ({'target_vx': args.drive_x, 'target_vy': args.drive_y}
                if getattr(args, 'fixed_drive', False) else {})
    obs = env.reset(**reset_kw)
    ep_reward = 0.0
    ep_len = 0
    ep_count = 0
    recent_rewards = deque(maxlen=20)
    best_eval_reward = -float('inf')
    log_interval = 1000

    # Windowed FPS tracking (excludes eval time)
    t_train_start = time.time()
    train_time_acc = 0.0
    t_window = time.time()
    steps_window = 0

    for step in range(args.total_steps):
        t_step_start = time.time()

        # Zero-action warmup: collect pure PID data as baseline
        if step < args.warmup_random:
            action = np.zeros(act_dim)
        else:
            action = agent.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        agent.buffer.push(obs, action, reward, next_obs, done)
        ep_reward += reward
        ep_len += 1

        # Update SAC after warmup; optionally pretrain the critic alone for
        # --actor-delay steps so the actor doesn't chase an untrained Q
        train_info = {}
        if step >= args.warmup_random:
            actor_on = step >= args.warmup_random + getattr(args, 'actor_delay', 0)
            train_info = agent.update(update_actor=actor_on)

        train_time_acc += time.time() - t_step_start
        steps_window += 1

        if done:
            recent_rewards.append(ep_reward)
            ep_count += 1
            obs = env.reset(**reset_kw)
            ep_reward = 0.0
            ep_len = 0
        else:
            obs = next_obs

        # Logging (windowed FPS, excludes eval time)
        if step > 0 and step % log_interval == 0:
            dt_window = time.time() - t_window
            fps = steps_window / max(dt_window, 1e-3)
            t_window = time.time()
            steps_window = 0
            avg_r = np.mean(recent_rewards) if recent_rewards else 0
            alpha_sac = train_info.get('alpha', 0)
            c_loss = train_info.get('critic_loss', 0)
            print(f"step={step:7d}  ep={ep_count:4d}  "
                  f"avg_r={avg_r:+7.2f}  a_env={env.alpha:.3f}  "
                  f"a_sac={alpha_sac:.3f}  c_loss={c_loss:.3f}  "
                  f"buf={len(agent.buffer):6d}  fps={fps:.0f}  "
                  f"terrain={env._current_terrain_label}")

        # Periodic evaluation + checkpoint (timing excluded from FPS)
        if step > 0 and step % args.eval_interval == 0:
            eval_reward = evaluate_quick(env, agent, n_episodes=3)
            print(f"  [EVAL] step={step}  mean_reward={eval_reward:+.2f}")
            agent.save(ckpt_dir / 'latest.pt')
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(ckpt_dir / 'best.pt')
                print(f"  [EVAL] New best: {best_eval_reward:+.2f}")
            # Reset FPS window after eval (eval time not counted)
            t_window = time.time()
            steps_window = 0

    # Final save
    agent.save(ckpt_dir / 'final.pt')
    total_time = time.time() - t_train_start
    print(f"\nTraining complete. {args.total_steps} steps, {ep_count} episodes, "
          f"{total_time:.0f}s total.")
    print(f"Checkpoints saved to {ckpt_dir}")


def evaluate_quick(env, agent, n_episodes=5):
    """Quick evaluation: deterministic policy on FIXED terrain seeds.

    Fixed seeds make eval scores comparable across checkpoints — with random
    terrain the +-150 episode variance makes best-checkpoint selection a
    lottery. The training RNG stream is restored afterwards."""
    saved_global_step = env.global_step
    rng_state = np.random.get_state()
    rewards = []
    for i in range(n_episodes):
        np.random.seed(7000 + i)
        obs = env.reset(target_vx=0.7, target_vy=0.0)
        ep_r = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            ep_r += r
        rewards.append(ep_r)
    env.global_step = saved_global_step
    np.random.set_state(rng_state)
    return np.mean(rewards)


# ======================================================================
# Full evaluation: PID vs PID+Residual
# ======================================================================

def _mix_residual_for_log(res, act_dim):
    """Mix structured residual to 6 motors (same logic as env._mix_residual)."""
    res_roll = res[0]
    res_pitch = res[1]
    res_yaw = res[2]
    res_x = res[3] if act_dim >= 5 else 0.0
    res_y = res[4] if act_dim >= 5 else 0.0
    return np.array([
        +res_yaw + res_y,    # M0
        -res_yaw + res_y,    # M1
        +res_roll,           # M2
        -res_roll,           # M3
        +res_pitch + res_x,  # M4
        -res_pitch + res_x,  # M5
    ])


def _run_eval_episode(env, agent, mode, target_vx, target_vy):
    """Run one eval episode and return log dict.

    Logs PID output, residual (structured), residual mixed to 6 motors,
    and final combined ctrl separately for decomposition analysis.
    """
    from control.run_baseline import get_ball_vel

    act_dim = env.act_dim
    env.global_step = 999999  # full alpha
    obs = env.reset(target_vx=target_vx, target_vy=target_vy)
    log = {
        'rpy': [],           # (N, 3) attitude angles in degrees
        'vel_err_x': [],     # (N,)  velocity error X
        'vel_err_y': [],     # (N,)  velocity error Y
        'pid_ctrl': [],      # (N, 6) PID-only motor output
        'residual': [],      # (N, act_dim) structured residual
        'residual_6': [],    # (N, 6) residual mixed to 6 motors
        'combined_ctrl': [], # (N, 6) final ctrl = clip(pid + residual_6)
    }

    done = False
    while not done:
        if mode == 'pid_residual':
            action = agent.select_action(obs, deterministic=True)
        else:
            action = np.zeros(act_dim)
        obs, r, done, info = env.step(action)

        imu = env.sim.get_imu()
        ball_vel = get_ball_vel(env.sim.model, env.sim.data, env.shell_id)

        pid_ctrl = info['pid_ctrl']
        res = info['residual']
        res6 = _mix_residual_for_log(res, act_dim)
        combined = np.clip(pid_ctrl + res6, -1.0, 1.0)

        log['rpy'].append(imu['euler'].copy())
        log['vel_err_x'].append(target_vx - ball_vel[0])
        log['vel_err_y'].append(target_vy - ball_vel[1])
        log['pid_ctrl'].append(pid_ctrl.copy())
        log['residual'].append(res.copy())
        log['residual_6'].append(res6.copy())
        log['combined_ctrl'].append(combined.copy())

    return {k: np.array(v) for k, v in log.items()}


def _compute_stats(log, skip_frac=0.25):
    """Compute summary stats from log, skipping initial transient."""
    n = max(1, int(len(log['rpy']) * skip_frac))
    rpy = log['rpy'][n:]
    vx_err = log['vel_err_x'][n:]
    vy_err = log['vel_err_y'][n:]
    ctrl = log['combined_ctrl'][n:]
    res = log['residual'][n:]

    # Attitude RMS (roll + pitch, in degrees)
    att_rms = np.sqrt(np.mean(rpy[:, 0] ** 2 + rpy[:, 1] ** 2))
    # Velocity tracking RMS
    vel_rms = np.sqrt(np.mean(vx_err ** 2 + vy_err ** 2))
    # Control smoothness: mean |Δctrl| per step (lower = smoother)
    d_ctrl = np.diff(ctrl, axis=0)
    ctrl_smooth = np.mean(np.sqrt(np.sum(d_ctrl ** 2, axis=1)))
    # Mean residual magnitude
    res_mag = np.mean(np.sqrt(np.sum(res ** 2, axis=1)))

    return {'att_rms': att_rms, 'vel_rms': vel_rms,
            'ctrl_smooth': ctrl_smooth, 'res_mag': res_mag}


def evaluate_full(args):
    """Compare PID-only vs PID+residual on velocity tracking."""
    act_dim = args.act_dim
    # Auto-detect obs_dim from checkpoint
    ckpt_info = peek_checkpoint(args.policy)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=act_dim)
    agent.load(args.policy)
    print(f"Loaded policy from {args.policy} (act_dim={act_dim}, "
          f"obs_dim={obs_dim}{', no_joint' if no_joint else ''})")

    scenarios = [
        {'name': 'vx=0.7', 'vx': 0.7, 'vy': 0.0},
        {'name': 'vy=0.5', 'vx': 0.0, 'vy': 0.5},
        {'name': 'diagonal', 'vx': 0.5, 'vy': 0.5},
    ]

    results = {}
    print(f"\n{'Scenario':12s} {'Mode':14s} {'att_rms':>8s} {'vel_rms':>8s} {'smooth':>8s} {'res_mag':>8s}")
    print("-" * 68)
    for sc in scenarios:
        for mode in ['pid_only', 'pid_residual']:
            env = SphericalRobotEnv(
                alpha_max=args.alpha_max if mode == 'pid_residual' else 0.0,
                warmup_steps=0,
                max_episode_steps=1000,
                randomize=False,
                act_dim=act_dim,
                unidirectional=args.unidirectional,
                terrain_prob=args.terrain_prob,
                roughness_range=(args.roughness_min, args.roughness_max),
                terrain_switch_interval=args.terrain_switch_interval,
                no_joint_obs=no_joint,
            )
            log = _run_eval_episode(env, agent, mode, sc['vx'], sc['vy'])
            key = f"{sc['name']}_{mode}"
            results[key] = log

            stats = _compute_stats(log)
            print(f"  {sc['name']:10s} {mode:14s} {stats['att_rms']:7.2f}° "
                  f"{stats['vel_rms']:7.3f}  {stats['ctrl_smooth']:7.4f}  "
                  f"{stats['res_mag']:7.4f}")

    if args.plot:
        plot_comparison(results, scenarios, act_dim)


def plot_comparison(results, scenarios, act_dim):
    """Generate detailed comparison plots: PID vs PID+Residual.

    Layout: n_scenarios rows x 5 columns
      Col 0: Attitude error (Roll, Pitch) — PID vs RL overlay
      Col 1: Velocity tracking error (vx, vy components)
      Col 2: Ctrl decomposition — PID output, residual, combined (one motor pair)
      Col 3: All residual components over time (3 or 5 depending on act_dim)
      Col 4: Control smoothness (|d(ctrl)/dt|) comparison
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    n_sc = len(scenarios)
    dt = 0.02  # PID dt
    fig, axes = plt.subplots(n_sc, 5, figsize=(26, 4.5 * n_sc))
    if n_sc == 1:
        axes = axes[np.newaxis, :]

    if act_dim == 3:
        res_labels = ['roll_diff', 'pitch_diff', 'yaw_diff']
        res_colors = ['tab:blue', 'tab:red', 'tab:green']
    else:
        res_labels = ['roll_diff', 'pitch_diff', 'yaw_diff', 'x_drive', 'y_drive']
        res_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

    for i, sc in enumerate(scenarios):
        pid_log = results[f"{sc['name']}_pid_only"]
        rl_log = results[f"{sc['name']}_pid_residual"]
        t_pid = np.arange(len(pid_log['rpy'])) * dt
        t_rl = np.arange(len(rl_log['rpy'])) * dt

        # ---- Col 0: Attitude (Roll, Pitch) ----
        ax = axes[i, 0]
        ax.plot(t_pid, pid_log['rpy'][:, 0], 'b--', alpha=0.5, lw=1, label='PID Roll')
        ax.plot(t_pid, pid_log['rpy'][:, 1], 'r--', alpha=0.5, lw=1, label='PID Pitch')
        ax.plot(t_rl, rl_log['rpy'][:, 0], 'b-', alpha=0.8, lw=1.2, label='RL Roll')
        ax.plot(t_rl, rl_log['rpy'][:, 1], 'r-', alpha=0.8, lw=1.2, label='RL Pitch')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'{sc["name"]}: Attitude Error')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ---- Col 1: Velocity tracking error ----
        ax = axes[i, 1]
        ax.plot(t_pid, pid_log['vel_err_x'], 'b--', alpha=0.5, lw=1, label='PID vx_err')
        ax.plot(t_pid, pid_log['vel_err_y'], 'r--', alpha=0.5, lw=1, label='PID vy_err')
        ax.plot(t_rl, rl_log['vel_err_x'], 'b-', alpha=0.8, lw=1.2, label='RL vx_err')
        ax.plot(t_rl, rl_log['vel_err_y'], 'r-', alpha=0.8, lw=1.2, label='RL vy_err')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_ylabel('Vel error (m/s)')
        ax.set_title(f'{sc["name"]}: Velocity Tracking Error')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ---- Col 2: Ctrl decomposition (M4 = +pitch + x_drive, most active for vx) ----
        ax = axes[i, 2]
        motor_idx = 4  # M4: pitch + x_drive
        motor_name = 'M4'
        # PID-only baseline ctrl for reference
        ax.plot(t_pid, pid_log['combined_ctrl'][:, motor_idx], 'k--',
                alpha=0.4, lw=0.8, label=f'PID-only {motor_name}')
        # RL run decomposition
        ax.plot(t_rl, rl_log['pid_ctrl'][:, motor_idx], 'tab:blue',
                alpha=0.6, lw=0.9, label='PID component')
        ax.plot(t_rl, rl_log['residual_6'][:, motor_idx], 'tab:orange',
                alpha=0.7, lw=0.9, label='RL residual')
        ax.plot(t_rl, rl_log['combined_ctrl'][:, motor_idx], 'tab:red',
                alpha=0.9, lw=1.2, label=f'Combined {motor_name}')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.2)
        ax.set_ylabel('ctrl')
        ax.set_title(f'{sc["name"]}: Ctrl Decomposition ({motor_name})')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ---- Col 3: All residual components over time ----
        ax = axes[i, 3]
        for j, (lbl, clr) in enumerate(zip(res_labels, res_colors)):
            ax.plot(t_rl, rl_log['residual'][:, j], color=clr,
                    alpha=0.8, lw=1, label=lbl)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_ylabel('Residual value')
        ax.set_title(f'{sc["name"]}: RL Residual Components ({act_dim}-dim)')
        ax.legend(fontsize=7, ncol=min(3, act_dim))
        ax.grid(True, alpha=0.3)

        # ---- Col 4: Control smoothness (|d(ctrl)/dt|) ----
        ax = axes[i, 4]
        window = 25  # 0.5s smoothing
        for label, data, ls, alpha_v in [('PID', pid_log, '--', 0.5), ('RL', rl_log, '-', 0.8)]:
            d_ctrl = np.diff(data['combined_ctrl'], axis=0) / dt
            jerk = np.sqrt(np.sum(d_ctrl ** 2, axis=1))
            kernel = np.ones(window) / window
            jerk_smooth = np.convolve(jerk, kernel, mode='valid')
            t_j = np.arange(len(jerk_smooth)) * dt
            ax.plot(t_j, jerk_smooth, ls, alpha=alpha_v, label=label)
        ax.set_ylabel('|d(ctrl)/dt|')
        ax.set_title(f'{sc["name"]}: Control Smoothness')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for c in range(5):
            axes[i, c].set_xlabel('Time (s)')

    mode_label = 'attitude-only' if act_dim == 3 else 'full'
    fig.suptitle(f'PID vs PID+Residual RL ({mode_label}, {act_dim}-dim)', fontsize=14, y=1.01)
    plt.tight_layout()
    out = Path(__file__).parent / 'eval_comparison.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)


# ======================================================================
# Viewer mode
# ======================================================================

def run_viewer(args):
    """Watch trained policy in MuJoCo viewer."""
    import mujoco.viewer

    act_dim = args.act_dim
    # Auto-detect obs_dim from checkpoint
    ckpt_info = peek_checkpoint(args.policy)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=act_dim)
    agent.load(args.policy)
    print(f"Loaded policy from {args.policy} (act_dim={act_dim}, "
          f"obs_dim={obs_dim}{', no_joint' if no_joint else ''})")

    env = SphericalRobotEnv(
        alpha_max=args.alpha_max,
        warmup_steps=0,
        max_episode_steps=999999,
        act_dim=act_dim,
        unidirectional=args.unidirectional,
        terrain_prob=args.terrain_prob,
        roughness_range=(args.roughness_min, args.roughness_max),
        terrain_switch_interval=args.terrain_switch_interval,
        no_joint_obs=no_joint,
        shell_inertia=getattr(args, 'shell_inertia', None),
        inertia_scales=getattr(args, '_inertia_scales', None),
        attitude_gain_scale=getattr(args, 'gain_scale', 1.0),
        attitude_gains=getattr(args, '_attitude_gains', None),
        mass_overrides=(getattr(args, '_sim_config', None) or {}).get('mass_overrides'),
        inertia_overrides=(getattr(args, '_sim_config', None) or {}).get('inertia_overrides'),
        pos_overrides=(getattr(args, '_sim_config', None) or {}).get('pos_overrides'),
        dof_damping_scale=(getattr(args, '_sim_config', None) or {}).get('dof_damping_scale'),
    )
    env.global_step = 999999  # full alpha

    obs = env.reset(target_vx=args.drive_x, target_vy=args.drive_y)

    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        step_i = 0
        while viewer.is_running():
            t0 = time.perf_counter()
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset(target_vx=args.drive_x, target_vy=args.drive_y)

            viewer.sync()
            step_i += 1

            if step_i % 25 == 0:
                imu = env.sim.get_imu()
                r, p, y = imu['euler']
                res_mag = np.linalg.norm(info['residual'])
                print(f"\rt={env.sim.time:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  "
                      f"res={res_mag:.4f}  alpha={info['alpha']:.3f}",
                      end="", flush=True)

            # Real-time sync
            dt_pid = 1.0 / env.pid_rate
            sl = dt_pid - (time.perf_counter() - t0)
            if sl > 0:
                time.sleep(sl)
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Residual RL training and evaluation')

    # Mode
    parser.add_argument('--eval', action='store_true', help='Run evaluation (PID vs PID+RL)')
    parser.add_argument('--viewer', action='store_true', help='Watch policy in viewer')
    parser.add_argument('--policy', type=str, default=None, help='Path to saved policy')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')

    # Training
    parser.add_argument('--total-steps', type=int, default=300000)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--episode-steps', type=int, default=500)
    parser.add_argument('--warmup-random', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--utd-ratio', type=int, default=2,
                        help='Critic updates per env step (update-to-data ratio)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Residual RL
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--warmup', type=int, default=5000,
                        help='Alpha warmup steps (default: matches warmup-random)')
    parser.add_argument('--act-dim', type=int, default=3, choices=[3, 5],
                        help='Action dim: 3=attitude-only, 5=attitude+drive (default: 3)')
    parser.add_argument('--unidirectional', action='store_true',
                        help='Use unidirectional motor mixing (one motor per pair, positive cmd only)')
    parser.add_argument('--no-joint-obs', action='store_true',
                        help='Remove joint angles from obs (IMU-only mode, obs_dim=11 vs 15). '
                             'Test whether RL uses gimbal state or just does PID compensation.')

    # Terrain randomization
    parser.add_argument('--terrain-prob', type=float, default=0.5,
                        help='Probability of rough terrain per episode (0=flat-only, default: 0.5)')
    parser.add_argument('--roughness-min', type=float, default=0.05,
                        help='Min roughness for rough episodes (default: 0.05)')
    parser.add_argument('--roughness-max', type=float, default=0.2,
                        help='Max roughness for rough episodes (default: 0.2)')
    parser.add_argument('--terrain-switch-interval', type=int, default=1,
                        help='Switch terrain every N episodes (default: 1)')

    # Inertia realism (audit §2.1) — verification experiments, non-overwriting
    parser.add_argument('--shell-inertia', type=float, default=None,
                        help='Override shell (Link3) diag inertia, kg*m^2 '
                             '(realistic thin shell ~0.0157). Default: committed value.')
    parser.add_argument('--ring-scale', type=float, default=1.0,
                        help='Multiply inner-ring (Link1,Link2) inertia by this factor '
                             '(inner rings dominate attitude dynamics). Default 1.0 = unchanged.')
    parser.add_argument('--gain-scale', type=float, default=1.0,
                        help='(legacy, crude) uniform scale of all attitude gains. Default 1.0.')
    parser.add_argument('--gains-json', type=str, default=None,
                        help='Path to JSON with a per-axis PID parameter set (proper re-tune) '
                             '{"roll":{"inner_kp":...},"pitch":{...},"yaw":{...}}. '
                             'This is the base controller initialized with the tuned params.')
    parser.add_argument('--sim-config-json', type=str, default=None,
                        help='Path to JSON with realistic sim config: keys mass_overrides, '
                             'inertia_overrides, pos_overrides, dof_damping_scale.')
    parser.add_argument('--ckpt-subdir', type=str, default=None,
                        help='Checkpoint subdir under checkpoints/ (avoid overwriting existing).')
    parser.add_argument('--yaw-reward', type=float, default=0.0,
                        help='Weight of |yaw error| in r_att (0 = legacy reward '
                             'with no heading term). The tuned-PID headroom is '
                             'concentrated in yaw at gimbal-lock passages, so '
                             'the residual needs this term to have a target.')
    parser.add_argument('--act-penalty', type=float, default=10.0,
                        help='Residual magnitude penalty coefficient (legacy 10 '
                             'makes any useful action net-negative; see env).')
    parser.add_argument('--smooth-penalty', type=float, default=2.0,
                        help='Residual smoothness penalty coefficient.')
    parser.add_argument('--target-entropy', type=float, default=None,
                        help='SAC target entropy (default -act_dim). Residual '
                             'RL wants lower (e.g. -6): the default keeps the '
                             'policy stddev large relative to the tiny residual '
                             'headroom and the noise erases the advantage.')
    parser.add_argument('--fixed-drive', action='store_true',
                        help='Train with fixed target velocity (--drive-x/-y) '
                             'instead of randomized targets (matches eval task).')
    parser.add_argument('--realism', action='store_true',
                        help='Train with the sensor/actuator realism pack '
                             '(IMU noise 0.5deg/0.02rad/s, 20 ms latency, '
                             '80 ms thrust lag) — policies trained without it '
                             'collapse under it at eval.')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='Stack the last N obs frames (temporal context; '
                             'needed to recover state under the realism pack).')
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-4)
    parser.add_argument('--actor-delay', type=int, default=0,
                        help='Critic-only pretraining steps after warmup before '
                             'actor updates start (shallows the early dip where '
                             'the actor chases an untrained Q).')
    parser.add_argument('--init-alpha', type=float, default=1.0,
                        help='Initial SAC entropy coefficient (default 1.0 = '
                             'legacy). Use ~0.05 on strong actuators so the '
                             'early entropy phase does not flood the buffer '
                             'with catastrophic residuals.')

    # Viewer / eval
    parser.add_argument('--drive-x', type=float, default=0.7)
    parser.add_argument('--drive-y', type=float, default=0.0)

    args = parser.parse_args()

    # Inner-ring inertia scaling → dict for the env/sim override
    args._inertia_scales = None
    if getattr(args, 'ring_scale', 1.0) and args.ring_scale != 1.0:
        args._inertia_scales = {'Link1': args.ring_scale, 'Link2': args.ring_scale}

    # Explicit tuned PID parameter set (proper re-tune) from JSON
    args._attitude_gains = None
    if getattr(args, 'gains_json', None):
        import json as _json
        with open(args.gains_json) as _f:
            args._attitude_gains = _json.load(_f)
        print(f"Loaded tuned PID gains from {args.gains_json}")

    # Realistic sim config (mass/inertia/pos/damping) from JSON
    args._sim_config = None
    if getattr(args, 'sim_config_json', None):
        import json as _json
        with open(args.sim_config_json) as _f:
            args._sim_config = _json.load(_f)
        print(f"Loaded realistic sim config from {args.sim_config_json}: "
              f"{list(args._sim_config.keys())}")

    if not args.policy:
        args.policy = str(CKPT_DIR / 'best.pt')

    if args.eval:
        evaluate_full(args)
    if args.viewer:
        run_viewer(args)
    if not args.eval and not args.viewer:
        train(args)


if __name__ == '__main__':
    main()
