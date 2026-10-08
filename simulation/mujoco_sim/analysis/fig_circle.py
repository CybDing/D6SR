#!/usr/bin/env python3
"""
Paper Fig. 5 (same visual form as the submitted version): circular-trajectory
tracking, (a) without attitude control vs (b) with PID + residual RL.

New conditions: calibrated plant + sensing/actuation realism, rough terrain
(default r = 0.2, the tuning/training severity). Circle radius 10 m, 0.5 m/s.

Usage (from simulation/mujoco_sim/):
  python analysis/fig_circle.py --policy rl/checkpoints/calib_v7b_nojoint_fs4/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.common import make_env, load_policy, policy_obs_layout, MEDIA
from paper_style import PAPER_RC, panel_caption

RADIUS = 10.0
SPEED = 0.5
KP_TRACK = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--gains', type=str,
                    default='config/tuned_gains_calibrated.json')
    ap.add_argument('--committed', action='store_true', default=False)
    ap.add_argument('--realism', action='store_true', default=True)
    ap.add_argument('--alpha-max', type=float, default=0.05)
    ap.add_argument('--force-max', type=float, default=None)
    ap.add_argument('--roughness', type=float, default=0.2)
    ap.add_argument('--terrain-res', type=int, default=400)
    ap.add_argument('--seconds', type=float, default=140.0)
    ap.add_argument('--seed', type=int, default=19)
    args = ap.parse_args()

    agent = load_policy(args.policy)
    fs, nj = policy_obs_layout(args.policy)
    panels = []
    # Driving convention for the circle (part of the task definition): the yaw
    # setpoint follows the path tangent and the drive command is rotated into
    # the current heading (yaw compensation), so propulsion always uses the
    # forward (pitch-pair) axis. Velocity-loop output limit scaled to the
    # calibrated motors (0.8 * 5 / 14.7 ~= 0.3). Episodes are NOT terminated
    # on capsize: the flip instant is marked and the run continues, so failed
    # strategies show their characteristic wandering (as in the original
    # paper's Fig. 5a) instead of a truncated arc.
    # Protocol: NO heading compensation anywhere — the drive allocation
    # assumes the attitude loop keeps the body aligned with the world frame
    # (paper Sec. IV-B). The commanded velocity is clipped to |v| <= SPEED
    # (sane tracker). Flips are marked (x) and the run continues: a controller
    # that recovers its attitude resumes tracking.
    for label, ag, fsn, njn, dis in (
            ("(a) Without attitude control", None, 1, False, True),
            ("(b) PID + residual RL", agent, fs, nj, False)):
        env = make_env(args, args.roughness, frame_stack=fsn, no_joint_obs=njn)
        np.random.seed(args.seed)
        obs = env.reset(target_vx=0.0, target_vy=0.0)
        env.controller.vel_pid_x.output_limit = 0.3
        env.controller.vel_pid_y.output_limit = 0.3
        if dis:
            env.controller.enabled = False        # fair baseline: no comp
        else:
            env.controller.yaw_compensate = True  # continuous compensation
        path, refs = [], []
        flip_idx = None
        over_cnt = 0     # sustained-flip: tilt > 60 deg for >= 0.2 s
        center = np.array([0.0, RADIUS])
        arc = 0.0
        for i in range(int(args.seconds * 50)):
            t = i / 50.0
            v_t = min(SPEED, 0.16 * t)            # 5 s speed ramp
            arc += v_t / 50.0
            th = arc / RADIUS
            p_ref = center + RADIUS * np.array([np.sin(th), -np.cos(th)])
            v_ref = v_t * np.array([np.cos(th), np.sin(th)])
            p = env.sim.data.xpos[env.shell_id][:2]
            v_cmd = v_ref + KP_TRACK * (p_ref - p)
            nv = np.linalg.norm(v_cmd)
            if nv > SPEED:
                v_cmd *= SPEED / nv               # tracker velocity clip
            env.controller.set_drive(vx=float(v_cmd[0]), vy=float(v_cmd[1]))
            if not dis:
                env.controller.yaw_pid.set_target(np.degrees(th))
                env.target_yaw = th               # keep policy obs consistent
            action = (ag.select_action(obs, deterministic=True)
                      if ag is not None else np.zeros(env.act_dim))
            obs, _, done, _ = env.step(action)
            path.append(p.copy()); refs.append(p_ref.copy())
            eul = env.sim.get_imu()['euler']
            over_cnt = over_cnt + 1 \
                if (abs(eul[0]) > 60 or abs(eul[1]) > 60) else 0
            if over_cnt >= 10 and flip_idx is None:
                flip_idx = i
                print(f"  {label}: flipped at t={t:.1f}s (run continues)")
        path, refs = np.asarray(path), np.asarray(refs)
        err = np.linalg.norm(path - refs, axis=1)
        print(f"{label}: mean err={err.mean():.2f} m  max={err.max():.2f} m")
        panels.append((label, path, err, flip_idx))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC)
    th = np.linspace(0, 2 * np.pi, 400)
    # authored at COLUMN_W so \includegraphics[width=\columnwidth] is a no-op
    # and the PAPER_RC point sizes reach the page unscaled (+ tight's crop)
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.35))
    for ax, (label, path, err, flip_idx) in zip(axes, panels):
        ax.plot(RADIUS * np.sin(th), RADIUS - RADIUS * np.cos(th), 'k--',
                lw=1.0, label='Reference')
        ax.plot(path[:, 0], path[:, 1], color='#3366cc', lw=0.9,
                label='Actual')
        ax.plot(*path[0], 'o', color='#33aa33', ms=6, label='Start')
        ax.plot(*path[-1], 's', color='#dd8800', ms=6, label='End')
        if flip_idx is not None:
            ax.plot(*path[flip_idx], 'x', color='#cc2222', ms=9,
                    markeredgewidth=2.5, label='Flip')
        stat = (rf"$\bar{{e}}$ = {err.mean():.2f} m" if err.mean() < 50 else
                "diverges after flip")
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_xlim(-RADIUS - 8, RADIUS + 8)
        ax.set_ylim(-8, 2 * RADIUS + 8)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        # two lines: at column width the two panels sit close, so a
        # single-line caption runs into its neighbour (the error term is
        # already shortened to $\bar e$)
        panel_caption(ax, f"{label}\n({stat})")
    # one shared legend for both panels (they plot the same series) — at
    # column width a per-panel legend eats a third of the axes
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(handles), loc='upper center',
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.7,
               handlelength=1.4, handletextpad=0.3, borderpad=0.25,
               framealpha=0.9, edgecolor='0.75', fancybox=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = MEDIA / 'fig5_circle_calibrated.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == '__main__':
    main()
