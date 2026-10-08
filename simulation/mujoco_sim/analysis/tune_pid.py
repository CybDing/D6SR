#!/usr/bin/env python3
"""
Coordinate-descent PID re-tune on the (calibrated) realistic plant.

Objective: total attitude RMS (roll+pitch+yaw) over rough-terrain drive
episodes, optionally under the sensor/actuator realism pack. Tunes
outer_kp / inner_kp / inner_kd per axis multiplicatively.

Usage (from simulation/mujoco_sim/):
  python analysis/tune_pid.py --realism --out config/tuned_gains_calibrated.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import run_episode, attitude_metrics

CFG_DIR = Path(__file__).resolve().parent.parent / 'config'

AXES = ('roll', 'pitch', 'yaw')
FIELDS = ('outer_kp', 'inner_kp', 'inner_kd')
FACTORS = (0.35, 0.5, 0.707, 1.0, 1.414, 2.0)


def evaluate(gains, args):
    """Total attitude RMS across seeds (lower is better)."""
    # write gains into a temp attitude_gains dict consumed by make_env via a
    # monkeypatched loader — simpler: build env manually here
    from rl.env import SphericalRobotEnv
    cfg = {k: v for k, v in json.load(open(CFG_DIR / 'realistic_config.json')).items()
           if not k.startswith('_')}
    realism = dict(imu_delay_steps=1, imu_angle_noise=0.5,
                   imu_gyro_noise=0.02, thrust_tau=0.08) if args.realism else {}
    env = SphericalRobotEnv(
        act_dim=3, unidirectional=True, max_episode_steps=10 ** 9,
        randomize=False, terrain_prob=1.0,
        roughness_range=(args.roughness, args.roughness),
        force_max=cfg.get('force_max', 5),
        mass_overrides=cfg.get('mass_overrides'),
        inertia_overrides=cfg.get('inertia_overrides'),
        pos_overrides=cfg.get('pos_overrides'),
        dof_damping_scale=cfg.get('dof_damping_scale'),
        joint_frictionloss=cfg.get('joint_frictionloss'),
        joint_damping=cfg.get('joint_damping'),
        attitude_gains=gains, **realism)
    env.global_step = 10 ** 9
    cost = 0.0
    for s in range(args.seeds):
        log = run_episode(env, None, seconds=args.seconds, seed=19 + s)
        m = attitude_metrics(log)
        c = m['att_rms']
        if len(log['t']) * env.steps_per_pid * env.dt_physics < args.seconds - 1:
            c += 200.0                      # flipped — heavy penalty
        cost += c
    return cost / args.seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--realism', action='store_true')
    ap.add_argument('--committed', action='store_true', default=False)  # for make_env compat
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--roughness', type=float, default=0.2)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--init', type=str,
                    default=str(CFG_DIR / 'tuned_gains_realistic.json'))
    ap.add_argument('--out', type=str,
                    default=str(CFG_DIR / 'tuned_gains_calibrated.json'))
    args = ap.parse_args()

    gains = {k: dict(v) for k, v in json.load(open(args.init)).items()
             if not k.startswith('_')}
    best = evaluate(gains, args)
    print(f"init cost = {best:.2f}   gains = {json.dumps(gains)}", flush=True)

    for p in range(args.passes):
        for axis in AXES:
            for field in FIELDS:
                base = gains[axis].get(field)
                if base is None:
                    continue
                for f in FACTORS:
                    if f == 1.0:
                        continue
                    trial = {a: dict(v) for a, v in gains.items()}
                    trial[axis][field] = base * f
                    c = evaluate(trial, args)
                    tag = ""
                    if c < best:
                        best, gains = c, trial
                        tag = "  <-- keep"
                    print(f"pass{p} {axis}.{field} x{f:<5} cost={c:7.2f} "
                          f"(best {best:7.2f}){tag}", flush=True)
        print(f"== pass {p} done: best={best:.2f} ==", flush=True)

    out = {'_comment': f"coordinate-descent re-tune on calibrated plant, "
                       f"realism={args.realism}, cost={best:.2f}"}
    out.update(gains)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f"saved {args.out}  (cost {best:.2f})")


if __name__ == '__main__':
    main()
