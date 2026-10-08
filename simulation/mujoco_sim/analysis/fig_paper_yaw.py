#!/usr/bin/env python3
"""
Regenerate the paper's Fig. 7 (yaw evolution while rolling over uneven
terrain) with the calibrated plant + sensing/actuation realism:

  (a) No attitude control          — unbounded yaw drift
  (b) Re-tuned cascaded PID        — bounded
  (c) Re-tuned PID + residual RL   — bounded, tighter

Mean ± 1 sigma bands over N seeds, 50 s episodes, paper-style layout.

Usage (from simulation/mujoco_sim/):
  python analysis/fig_paper_yaw.py --policy rl/checkpoints/calib_v6_fs4/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import (make_env, run_episode, load_policy,
                             policy_obs_layout, MEDIA)
from paper_style import PAPER_RC, panel_caption


def collect(env, agent, seeds, seconds, disable_attitude=False):
    n = int(seconds * 50)
    runs = np.full((seeds, n), np.nan)
    for s in range(seeds):
        log = run_episode(env, agent, seconds=seconds, seed=19 + s,
                          disable_attitude=disable_attitude,
                          stop_on_done=False)   # flips don't end figure runs
        y = log['rpy_err'][:, 2]
        # plot the RAW accumulated yaw (unwrapped), as in the original paper —
        # wrapped values fold drift into +-180 and hide it
        y = np.degrees(np.unwrap(np.radians(y)))
        runs[s, :len(y)] = y
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--gains', type=str,
                    default='config/tuned_gains_calibrated.json')
    ap.add_argument('--committed', action='store_true', default=False)
    ap.add_argument('--realism', action='store_true', default=True)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--seconds', type=float, default=50.0)
    ap.add_argument('--roughness', type=float, default=0.3)
    ap.add_argument('--terrain-res', type=int, default=400)
    ap.add_argument('--alpha-max', type=float, default=0.05,
                    help='must match the policy training value')
    args = ap.parse_args()

    agent = load_policy(args.policy)
    fs, nj = policy_obs_layout(args.policy)
    tiers = []
    for label, ag, dis in (('(a) Without attitude control', None, True),
                           ('(b) Cascaded PID (tuned at r = 0.2)', None, False),
                           ('(c) PID + residual RL (trained r ≤ 0.2)', agent, False)):
        env = make_env(args, args.roughness,
                       frame_stack=fs if ag is not None else 1,
                       no_joint_obs=nj if ag is not None else False)
        if not dis:
            env.controller.yaw_compensate = True   # protective heading comp
        runs = collect(env, ag, args.seeds, args.seconds, dis)
        tiers.append((label, runs))
        m = np.nanstd(runs)
        print(f"{label}: std={m:.1f} deg  max|yaw|={np.nanmax(np.abs(runs)):.1f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC)
    colors = ['#cc3333', '#3366cc', '#33aa33']
    t = np.arange(int(args.seconds * 50)) / 50.0
    # authored at COLUMN_W so \includegraphics[width=\columnwidth] is a no-op
    # and the PAPER_RC point sizes reach the page unscaled (+ tight's crop)
    fig, axes = plt.subplots(3, 1, figsize=(3.54, 5.3), sharex=True,
                             gridspec_kw={'hspace': 0.45})
    axes[-1].set_xlabel('Time (s)')
    for ax, (label, runs), c in zip(axes, tiers, colors):
        mu = np.nanmean(runs, axis=0)
        sd = np.nanstd(runs, axis=0)
        ax.plot(t, mu, color=c, lw=1.0, label=r'Mean $\theta_y$ ($\mu$)')
        ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.25,
                        label=r'$\pm 1\sigma$')
        stats = (f"std={np.nanstd(runs):.1f}°, "
                 f"max={np.nanmax(np.abs(runs)):.1f}°")
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.set_ylabel(r'$\theta_y$ (°)')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.25)
        panel_caption(ax, f"{label}  ({stats})")
    out = MEDIA / 'fig7_yaw_evolution_calibrated.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == '__main__':
    main()
