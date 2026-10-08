#!/usr/bin/env python3
"""
Shared helpers for the paper-figure and PID-tuning scripts in analysis/.

Builds the evaluation environment (calibrated plant + optional sensor/actuator
realism pack + tuned PID gains), loads residual-RL checkpoints, and rolls
episodes with a per-control-step attitude log.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.env import SphericalRobotEnv, wrap_angle

REPO = Path(__file__).resolve().parents[3]
MEDIA = REPO / 'docs' / 'media'
DATA = REPO / 'data'
CFG_DIR = Path(__file__).resolve().parent.parent / 'config'


def load_json(p):
    d = json.load(open(p))
    return {k: v for k, v in d.items() if not k.startswith('_')}


def make_env(args, roughness, frame_stack=1, no_joint_obs=False):
    cfg = load_json(CFG_DIR / 'realistic_config.json') if not args.committed else {}
    gains_path = getattr(args, 'gains', None) or (CFG_DIR / 'tuned_gains_realistic.json')
    gains = load_json(gains_path) if not args.committed else None
    # sensor/actuator realism pack (documented assumptions):
    # 1 PID period (20 ms) loop latency, 0.5 deg attitude noise,
    # 0.02 rad/s gyro noise, 80 ms ESC+prop thrust time constant
    realism = dict(imu_delay_steps=1, imu_angle_noise=0.5,
                   imu_gyro_noise=0.02, thrust_tau=0.08) \
        if getattr(args, 'realism', False) else {}
    env = SphericalRobotEnv(
        act_dim=3,
        unidirectional=True,
        max_episode_steps=10 ** 9,
        randomize=False,
        alpha_max=getattr(args, 'alpha_max', 0.15),
        terrain_prob=1.0 if roughness > 0 else 0.0,
        roughness_range=(roughness, roughness),
        force_max=(getattr(args, 'force_max', None) or cfg.get('force_max', 5)),
        mass_overrides=cfg.get('mass_overrides'),
        inertia_overrides=cfg.get('inertia_overrides'),
        pos_overrides=cfg.get('pos_overrides'),
        dof_damping_scale=cfg.get('dof_damping_scale'),
        joint_frictionloss=cfg.get('joint_frictionloss'),
        joint_damping=cfg.get('joint_damping'),
        attitude_gains=gains,
        frame_stack=frame_stack,
        no_joint_obs=no_joint_obs,
        terrain_kwargs=({'nrow': args.terrain_res, 'ncol': args.terrain_res}
                        if getattr(args, 'terrain_res', None) else None),
        **realism,
    )
    env.global_step = 10 ** 9          # full residual scale (alpha_max)
    return env


def policy_obs_layout(path):
    """Infer (frame_stack, no_joint_obs) from a checkpoint's obs_dim.

    Frames are 15-dim (with joint angles) or 11-dim (--no-joint-obs)."""
    if not path:
        return 1, False
    from rl.sac import peek_checkpoint
    d = peek_checkpoint(path)['obs_dim']
    if d % 15 == 0:
        return d // 15, False
    if d % 11 == 0:
        return d // 11, True
    raise ValueError(f"cannot infer obs layout from obs_dim={d}")


def load_policy(path, act_dim=3):
    if not path:
        return None
    from rl.sac import SAC, peek_checkpoint
    obs_dim = peek_checkpoint(path)['obs_dim']
    agent = SAC(obs_dim=obs_dim, act_dim=act_dim)
    agent.load(path)
    print(f"loaded policy {path} (obs_dim={obs_dim})")
    return agent


def yaw_err_deg(env, imu):
    return np.degrees(wrap_angle(env.target_yaw - np.radians(imu['euler'][2])))


def run_episode(env, agent=None, seconds=50.0, drive=(0.7, 0.0), seed=19,
                disable_attitude=False, stop_on_done=True):
    """Roll for `seconds`; returns per-control-step log arrays
    (t, rpy_err [deg], res).

    stop_on_done=False keeps stepping after a flip (figure runs: the flip is
    part of the story, not the end of it); tuning callers should keep the
    default."""
    np.random.seed(seed)
    obs = env.reset(target_vx=drive[0], target_vy=drive[1])
    if disable_attitude:
        env.controller.enabled = False
    steps = int(seconds * env.pid_rate) if hasattr(env, 'pid_rate') else \
        int(seconds / (env.steps_per_pid * env.dt_physics))
    log = {k: [] for k in ('t', 'rpy_err', 'res')}
    for i in range(steps):
        action = (agent.select_action(obs, deterministic=True)
                  if agent is not None else np.zeros(env.act_dim))
        obs, _, done, info = env.step(action)
        imu = env.sim.get_imu()
        log['t'].append(env.episode_step * env.steps_per_pid * env.dt_physics)
        r, p, _ = imu['euler']
        log['rpy_err'].append([r, p, yaw_err_deg(env, imu)])
        log['res'].append(info['residual'].copy())
        if done and stop_on_done:
            print(f"  episode terminated early at t={log['t'][-1]:.1f}s (flip)")
            break
    return {k: np.asarray(v) for k, v in log.items()}


def attitude_metrics(log):
    err = log['rpy_err']
    m = dict(
        roll_rms=float(np.sqrt((err[:, 0] ** 2).mean())),
        pitch_rms=float(np.sqrt((err[:, 1] ** 2).mean())),
        yaw_rms=float(np.sqrt((err[:, 2] ** 2).mean())),
        yaw_max=float(np.abs(err[:, 2]).max()),
        res_mag=float(np.abs(log['res']).mean()),
    )
    m['att_rms'] = m['roll_rms'] + m['pitch_rms'] + m['yaw_rms']
    return m
