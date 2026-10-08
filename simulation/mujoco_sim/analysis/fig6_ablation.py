#!/usr/bin/env python3
"""
Paper Fig. 6 — generated with the ORIGINAL paper plotting code
(benchmark_terrain_compare.plot_comparison_ablation), fed with runs from the
calibrated plant + sensing/actuation realism.

Protocol: one fixed terrain (seed 19), 5 runs per tier differing in the
sensor-noise realization; straight line at 0.5 m/s over the original paper's
fine terrain (600x600 cells over +-100 m, seed-19 master). Controlled tiers
use heading-compensated drive (protective, see revision guide); the
no-attitude tier does not (fair baseline). Flips are marked (x), runs
continue. Error unit: m.

Usage (from simulation/mujoco_sim/):
  python analysis/fig6_ablation.py --policy rl/checkpoints/calib_v7b_nojoint_fs4/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import make_env, load_policy, policy_obs_layout, MEDIA
from benchmark_terrain_compare import plot_comparison_ablation

SPEED = 1
TERRAIN_SEED = 42


class HfieldTerrain:
    """Adapter exposing the env's actual heightfield with the Terrain API
    that plot_comparison_ablation expects."""

    def __init__(self, env):
        m = env.sim.model
        hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
        self.nrow = int(m.hfield_nrow[hid])
        self.ncol = int(m.hfield_ncol[hid])
        self.size_x = float(m.hfield_size[hid][0])
        self.size_y = float(m.hfield_size[hid][1])
        self.elevation_max = float(m.hfield_size[hid][2])
        self._h = np.array(m.hfield_data[:self.nrow * self.ncol]).reshape(
            self.nrow, self.ncol)

    def generate_heights(self):
        return self._h


def run_once(env, agent, seconds, noise_seed, disable_attitude, speed):
    # Protocol matches the original benchmark: yaw-compensated drive for the
    # controlled tiers; the no-attitude tier drives without it (fair baseline,
    # audit 4.5). Without compensation a large yaw error turns the velocity
    # loop into positive feedback and runs away — not a controller property.
    # fixed terrain: the terrain sample is drawn from the global RNG inside
    # reset(); seed it identically for every run, then switch to the per-run
    # seed so only the sensing-noise realization differs.
    np.random.seed(TERRAIN_SEED)
    obs = env.reset(target_vx=speed, target_vy=0.0)
    np.random.seed(noise_seed)
    # per-run variation: gimbal initial configuration (same +-0.3 rad
    # protocol as training) + the sensing-noise realization
    env.sim.data.qpos[7:10] += np.random.uniform(-0.3, 0.3, 3)
    env.sim.data.qvel[:] = 0
    mujoco.mj_forward(env.sim.model, env.sim.data)
    if disable_attitude:
        env.controller.enabled = False        # fair baseline: no compensation
    else:
        env.controller.yaw_compensate = True  # continuous heading compensation
    p0 = env.sim.data.xpos[env.shell_id][:2].copy()
    pos, err, ts = [], [], []
    flip_pos = None
    over_cnt = 0        # sustained-flip detector: tilt > 60 deg for >= 0.2 s
    p_prev = p0.copy()
    for i in range(int(seconds * 50)):
        t = i / 50.0
        action = (agent.select_action(obs, deterministic=True)
                  if agent is not None else np.zeros(env.act_dim))
        obs, _, done, _ = env.step(action)
        p = env.sim.data.xpos[env.shell_id][:2]
        # truncate once the post-flip runaway leaves any physically
        # plausible regime (25 m/s): beyond that the contact solver output
        # is numerical junk (measured: QACC blowups, 45 m/step teleports
        # causing the vertical segments in earlier error curves)
        if np.linalg.norm(p - p_prev) > 0.5:
            break
        p_prev = p.copy()
        p_ref = p0 + np.array([speed * t, 0.0])
        pos.append(p.copy())
        err.append(float(np.linalg.norm(p - p_ref)))   # m
        ts.append(t)
        eul = env.sim.get_imu()['euler']
        over_cnt = over_cnt + 1 if (abs(eul[0]) > 60 or abs(eul[1]) > 60) \
            else 0
        if over_cnt >= 10 and flip_pos is None:   # sustained flip (0.2 s)
            flip_pos = p.copy()
    # reference = the full commanded course (speed x duration), independent
    # of where the run was truncated — the dashed line must match the true
    # expected rolling distance
    t_full = np.arange(int(seconds * 50)) / 50.0
    ref = np.column_stack([p0[0] + speed * t_full,
                           np.full(len(t_full), p0[1])])
    return dict(pos=np.asarray(pos), error=np.asarray(err),
                t=np.asarray(ts), ref=ref, flip_pos=flip_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--gains', type=str,
                    default='config/tuned_gains_calibrated.json')
    ap.add_argument('--committed', action='store_true', default=False)
    ap.add_argument('--realism', action='store_true', default=True)
    ap.add_argument('--alpha-max', type=float, default=0.05)
    ap.add_argument('--force-max', type=float, default=None)
    ap.add_argument('--roughness', type=float, default=0.3)
    ap.add_argument('--terrain-res', type=int, default=400)
    ap.add_argument('--seconds', type=float, default=65.0)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--runs', type=int, default=5)
    args = ap.parse_args()

    agent = load_policy(args.policy)
    fs, nj = policy_obs_layout(args.policy)
    tiers, terrain = [], None
    for label, ag, fsn, njn, dis in (
            ('(a) Without attitude control', None, 1, False, True),
            ('(b) Cascaded PID', None, 1, False, False),
            ('(c) PID + residual RL', agent, fs, nj, False)):
        env = make_env(args, args.roughness, frame_stack=fsn, no_joint_obs=njn)
        logs = []
        for r in range(args.runs):
            log = run_once(env, ag, args.seconds, 100 + r, dis,
                           args.speed)
            logs.append(log)
            flip = '  [flip]' if log['flip_pos'] is not None else ''
            print(f"  {label} run{r+1}: mean err "
                  f"{log['error'].mean():.2f} m{flip}", flush=True)
        if terrain is None:
            terrain = HfieldTerrain(env)
        tiers.append((logs, label,
                      label.split(') ')[1] if ') ' in label else label))

    course = args.speed * args.seconds
    out_pdf, means = plot_comparison_ablation(
        tiers, terrain, args, err_unit='m',
        out_name='fig6_terrain_ablation_calibrated.pdf',
        bounds_cap=(-12.0, course + 6.0, 25.0), flip_style=True,
        unify_y=(1, 2), single_column=True)
    # also copy into docs/media
    import shutil
    dst = MEDIA / 'fig6_terrain_ablation_calibrated.pdf'
    shutil.copy(out_pdf, dst)
    print(f"copied to {dst}")
    print("tier means (m):", [round(m, 2) for m in means])


if __name__ == '__main__':
    main()
