#!/usr/bin/env python3
"""
Benchmark: multi-start terrain comparison for IEEE paper figure.

Runs N multi-start trials on rough terrain, comparing:
  (a) No attitude control (drive only)
  (b) PID + Residual RL agent

Produces a 4-panel PDF figure:
  Top row:    XY trajectory plots with terrain heightmap overlay
  Bottom row: Tracking error mean +/- std bands

Usage:
    mjpython benchmark_terrain_compare.py
    mjpython benchmark_terrain_compare.py --multi-start 6 --roughness 0.2
    mjpython benchmark_terrain_compare.py --speed 0.8 --line-length 40
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from control.trajectory import LineTrajectory, TrajectoryTracker
from terrain import RoughTerrain
from rl.env import SphericalRobotEnv, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from paper_style import PAPER_RC, panel_caption


# ── Simulation runners ──────────────────────────────────────────────────


def run_baseline_traj(controller, duration, terrain, tracker, mu_roll,
                      force_max, start_offset=(0.0, 0.0),
                      attitude_enabled=False):
    """Trajectory tracking on terrain.

    attitude_enabled=False → drive only (no attitude PID), panel (a).
    attitude_enabled=True  → PID attitude control, no RL residual (PID-only tier).
    """
    sim = RobotSimulation(mu_roll=mu_roll, force_max=force_max, terrain=terrain)
    sim.reset()

    if start_offset[0] != 0.0 or start_offset[1] != 0.0:
        sim.data.qpos[0] = float(start_offset[0])
        sim.data.qpos[1] = float(start_offset[1])
        mujoco.mj_forward(sim.model, sim.data)

    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / controller.pid_rate / dt)))
    total = int(duration / dt)
    shell_id = sim._shell_body_id

    log = {'t': [], 'pos': [], 'ref': [], 'error': []}
    ctrl_total = np.zeros(6)

    # Settle
    for _ in range(int(0.5 / dt)):
        sim.step()

    controller.reset()
    tracker.reset()
    controller.set_target(roll=0, pitch=0, yaw=0)
    controller.enabled = attitude_enabled

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
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)

        if i % int(2.0 / dt) == 0:
            traj_t = sim.time - t_start
            ball_pos = sim.data.xpos[shell_id][:2]
            err = log['error'][-1] if log['error'] else 0
            print(f"\r    [baseline] t={traj_t:5.1f}s  err={err:.3f}m  "
                  f"pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    controller.enabled = True
    return {k: np.array(v) for k, v in log.items()}


def run_rl_traj(env, agent, controller, duration, terrain, tracker,
                start_offset=(0.0, 0.0)):
    """PID + RL residual, trajectory tracking on terrain."""
    sim = RobotSimulation(
        mu_roll=env.mu_roll, force_max=env.force_max, terrain=terrain)
    sim.reset()

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

    log = {'t': [], 'pos': [], 'ref': [], 'error': []}
    ctrl_total = np.zeros(6)
    residual = np.zeros(env.act_dim)

    # Settle
    for _ in range(int(0.5 / dt)):
        sim.step()

    controller.reset()
    tracker.reset()
    controller.set_target(roll=0, pitch=0, yaw=0)
    env.target_yaw = 0.0
    controller.enabled = True

    # Pre-stabilize 1s
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

    tracker.reset()
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

        if i % 20 == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            traj_t = sim.time - t_start
            ref_x, ref_y = tracker.traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            log['t'].append(traj_t)
            log['pos'].append(ball_pos.copy())
            log['ref'].append(np.array([ref_x, ref_y]))
            log['error'].append(err)

        if i % int(2.0 / dt) == 0:
            traj_t = sim.time - t_start
            ball_pos = sim.data.xpos[shell_id][:2]
            err = log['error'][-1] if log['error'] else 0
            print(f"\r    [PID+RL]  t={traj_t:5.1f}s  err={err:.3f}m  "
                  f"pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_comparison(baseline_logs, rl_logs, terrain, args):
    """IEEE 4-panel figure: trajectory + error, baseline vs PID+RL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    n_runs = len(baseline_logs)
    ref = baseline_logs[0]['ref']
    ref_y_center = float(np.mean(ref[:, 1]))

    # ── Per-panel axis bounds ──
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

    # ── Terrain heightmap ──
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
    h_mm = h_crop * 1000
    h_vmin, h_vmax = float(h_mm.min()), float(h_mm.max())

    # ── Figure layout ──
    fig = plt.figure(figsize=(7.16, 5.2))

    gs_top = gridspec.GridSpec(
        2, 1, figure=fig,
        left=0.08, right=0.87,
        bottom=0.38, top=0.97,
        hspace=0.45)

    cbar_ax = fig.add_axes([0.895, 0.38, 0.020, 0.59])

    gs_bot = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.08, right=0.96,
        bottom=0.07, top=0.28,
        wspace=0.35)

    ax_bl = fig.add_subplot(gs_top[0, 0])
    ax_rl = fig.add_subplot(gs_top[1, 0])
    ax_ebl = fig.add_subplot(gs_bot[0, 0])
    ax_erl = fig.add_subplot(gs_bot[0, 1])

    cmap_tab10 = plt.cm.tab10

    # ── Trajectory panels ──
    def _draw_traj_panel(ax, logs, title, x0, x1, y0, y1):
        ax.pcolormesh(Xg, Yg, h_mm, cmap='YlGnBu', alpha=0.40,
                      shading='auto', vmin=h_vmin, vmax=h_vmax)
        if np.ptp(h_mm) > 1e-3:
            n_levels = min(8, max(3, int(np.ptp(h_mm) / 5)))
            ax.contour(Xg, Yg, h_mm, levels=n_levels, colors='k',
                       linewidths=0.2, alpha=0.2)
        ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.2,
                label='Reference', zorder=5)
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            pos = log['pos']
            ax.plot(pos[:, 0], pos[:, 1], color=clr, linewidth=1.0,
                    alpha=0.85, label=f'Run {idx+1}', zorder=6)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=clr, markersize=3,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=7)
            ax.plot(pos[-1, 0], pos[-1, 1], 's', color=clr, markersize=2.5,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('auto')
        ax.set_ylabel('Y (m)')
        ax.set_title(title, fontweight='bold', pad=4)
        ax.legend(loc='upper left', fontsize=5.5, ncol=min(n_runs + 1, 6),
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
                  borderpad=0.3, framealpha=0.85, edgecolor='0.8')
        ax.grid(True, alpha=0.15, linewidth=0.4)

    _draw_traj_panel(ax_bl, baseline_logs, '(a) No attitude control',
                     bl_x0, bl_x1, bl_y0, bl_y1)
    _draw_traj_panel(ax_rl, rl_logs, '(b) PID + Residual RL agent',
                     rl_x0, rl_x1, rl_y0, rl_y1)
    ax_rl.set_xlabel('X (m)')

    # ── Shared terrain colorbar ──
    norm = Normalize(vmin=h_vmin, vmax=h_vmax)
    sm = ScalarMappable(cmap='YlGnBu', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Terrain height (mm)', fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6)

    # ── Error panels ──
    def _draw_error_panel(ax, logs, title):
        all_t, all_err = [], []
        max_t = 0
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            ax.plot(log['t'], log['error'], color=clr, alpha=0.30,
                    linewidth=0.5)
            all_t.append(log['t'])
            all_err.append(log['error'])
            max_t = max(max_t, log['t'][-1])

        t_common = np.linspace(0, max_t, 500)
        interp_err = np.array([
            np.interp(t_common, ta, ea) for ta, ea in zip(all_t, all_err)])
        mean_err = np.mean(interp_err, axis=0)
        std_err = np.std(interp_err, axis=0)
        overall_mean = float(np.mean(mean_err))

        ax.fill_between(t_common, np.maximum(mean_err - std_err, 0),
                        mean_err + std_err, color='gray', alpha=0.20)
        ax.plot(t_common, mean_err, 'k-', linewidth=1.0, alpha=0.85,
                label=f'Mean = {overall_mean:.3f} m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title(title, fontweight='bold', pad=4)
        ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9,
                  edgecolor='0.8', borderpad=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.4)

    _draw_error_panel(ax_ebl, baseline_logs,
                      '(c) Tracking error \u2014 No attitude control')
    _draw_error_panel(ax_erl, rl_logs,
                      '(d) Tracking error \u2014 PID + RL agent')

    # ── Save ──
    out_dir = Path(__file__).parent / 'plots'
    out_dir.mkdir(exist_ok=True)
    tag = f"rough{args.roughness}_{n_runs}runs"
    out_pdf = out_dir / f'benchmark_terrain_compare_{tag}.pdf'
    fig.savefig(str(out_pdf), format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_pdf}")


def plot_comparison_ablation(tiers, terrain, args, err_unit='m',
                             out_name='benchmark_terrain_compare_new.pdf',
                             bounds_cap=None, flip_style=False,
                             footnote=None, unify_y=None,
                             err_row_caption=None, single_column=False):
    """Three-tier ablation figure: No-control / PID-only / PID+RL.

    tiers : list of (logs, traj_title, err_title) — 3 entries, top→bottom.
    Trajectory panels stacked on top, tracking-error panels in a row below.
    err_unit: 'm' or 'cm' — log['error'] must already be in that unit.
    Optional log keys: 'flip_pos' (2,) marks the flip instant with an x.
    flip_style: aligned x-axes + two-class error panels
    (Mean (not flip) / Mean (flip runs)); trajectories stay per-run colored.
    err_row_caption: caption the error row as a single sub-figure instead of
    titling each panel with its tier name (which just repeats (a)-(c) above).
    single_column: lay out for an IEEE \\columnwidth (3.5 in) slot — only the
    trajectory panels are drawn, sharing one per-run legend. The error row is
    omitted because at 3.5 in each error legend is wider than its own panel;
    tier means are still computed and returned. err_row_caption is ignored.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    plt.rcParams.update({
        **PAPER_RC,
        'lines.linewidth': 1.2, 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    n_runs = len(tiers[0][0])
    ref = tiers[0][0][0]['ref']
    ref_y_center = float(np.mean(ref[:, 1]))

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

    bounds = [_panel_xy_bounds(logs) for logs, _, _ in tiers]
    if flip_style:
        # align x across all panels using the longest reference (truncated
        # runs carry shortened refs); y stays per panel
        all_ref_x = np.concatenate([log['ref'][:, 0]
                                    for logs, _, _ in tiers for log in logs])
        x_lo_c = all_ref_x.min() - 3.0
        x_hi_c = all_ref_x.max() + 4.0
        bounds = [(x_lo_c, x_hi_c, b[2], b[3]) for b in bounds]
    if unify_y:
        # share identical y-limits across the given tier indices
        y_lo = min(bounds[i][2] for i in unify_y)
        y_hi = max(bounds[i][3] for i in unify_y)
        bounds = [(b[0], b[1], y_lo, y_hi) if i in unify_y else b
                  for i, b in enumerate(bounds)]
    if bounds_cap is not None:
        xlo_c, xhi_c, yh_c = bounds_cap
        bounds = [(max(b[0], xlo_c), min(b[1], xhi_c),
                   max(b[2], ref_y_center - yh_c),
                   min(b[3], ref_y_center + yh_c)) for b in bounds]

    heights_m = terrain.generate_heights() * terrain.elevation_max
    cr, cc = terrain.nrow // 2, terrain.ncol // 2
    heights_m -= heights_m[cr, cc]
    x_coords = np.linspace(-terrain.size_x, terrain.size_x, terrain.ncol)
    y_coords = np.linspace(-terrain.size_y, terrain.size_y, terrain.nrow)
    xc_lo = min(b[0] for b in bounds); xc_hi = max(b[1] for b in bounds)
    yc_lo = min(b[2] for b in bounds); yc_hi = max(b[3] for b in bounds)
    xi_min = max(0, np.searchsorted(x_coords, xc_lo) - 1)
    xi_max = min(len(x_coords), np.searchsorted(x_coords, xc_hi) + 1)
    yi_min = max(0, np.searchsorted(y_coords, yc_lo) - 1)
    yi_max = min(len(y_coords), np.searchsorted(y_coords, yc_hi) + 1)
    Xg, Yg = np.meshgrid(x_coords[xi_min:xi_max], y_coords[yi_min:yi_max])
    h_mm = heights_m[yi_min:yi_max, xi_min:xi_max] * 1000
    h_vmin, h_vmax = float(h_mm.min()), float(h_mm.max())

    # captions sit below each panel, so every row reserves space under it
    if single_column:
        # authored at \columnwidth so LaTeX does not scale (and shrink) the
        # text; top strip is left free for the shared per-run legend.
        # hspace is wide because each caption hangs under its own panel: the
        # gap has to clear the caption AND still leave more air below it than
        # above, or the caption reads as a title for the panel underneath.
        fig = plt.figure(figsize=(3.55, 4.35))
        gs_top = gridspec.GridSpec(3, 1, figure=fig, left=0.145, right=0.80,
                                   bottom=0.145, top=0.95, hspace=0.86)
        cbar_ax = fig.add_axes([0.825, 0.145, 0.035, 0.805])
        gs_bot = None
    else:
        fig = plt.figure(figsize=(7.16, 8.0))
        gs_top = gridspec.GridSpec(3, 1, figure=fig, left=0.08, right=0.87,
                                   bottom=0.34, top=0.978, hspace=0.52)
        cbar_ax = fig.add_axes([0.895, 0.34, 0.020, 0.638])
        gs_bot = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.96,
                                   bottom=0.105, top=0.24, wspace=0.40)
    cmap_tab10 = plt.cm.tab10

    def _draw_traj_panel(ax, logs, title, b, show_x):
        ax.pcolormesh(Xg, Yg, h_mm, cmap='YlGnBu', alpha=0.40,
                      shading='auto', vmin=h_vmin, vmax=h_vmax,
                      rasterized=True)
        if np.ptp(h_mm) > 1e-3:
            n_levels = min(8, max(3, int(np.ptp(h_mm) / 5)))
            ax.contour(Xg, Yg, h_mm, levels=n_levels, colors='k',
                       linewidths=0.2, alpha=0.2)
        ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.2,
                label='Reference', zorder=5)
        for idx, log in enumerate(logs):
            clr = cmap_tab10(idx % 10)
            pos = log['pos']
            ax.plot(pos[:, 0], pos[:, 1], color=clr, linewidth=1.0,
                    alpha=0.85, label=f'Run {idx+1}', zorder=6)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=clr, markersize=3,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=7)
            ax.plot(pos[-1, 0], pos[-1, 1], 's', color=clr, markersize=2.5,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=7)
            if log.get('flip_pos') is not None:
                ax.plot(log['flip_pos'][0], log['flip_pos'][1], 'x',
                        color=clr, markersize=6, markeredgewidth=1.8,
                        zorder=8)
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3])
        ax.set_aspect('auto'); ax.set_ylabel('Y (m)')
        if not single_column:
            # at column width this 6-entry legend is wider than the panel;
            # one shared figure legend is added below instead
            ax.legend(loc='upper left', fontsize=5.5, ncol=min(n_runs + 1, 6),
                      handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
                      borderpad=0.3, framealpha=0.85, edgecolor='0.8')
        ax.grid(True, alpha=0.15, linewidth=0.4)
        if show_x:
            ax.set_xlabel('X (m)')
        if not single_column:
            panel_caption(ax, title)   # else: captioned once means are known

    top_axes = []
    for i, (logs, traj_title, _) in enumerate(tiers):
        ax = fig.add_subplot(gs_top[i, 0])
        _draw_traj_panel(ax, logs, traj_title, bounds[i], show_x=(i == 2))
        top_axes.append(ax)

    if single_column:
        # every tier plots the same series, so one legend serves all three
        handles, labels = top_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=min(n_runs + 1, 6), loc='upper center',
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
                   handletextpad=0.3, columnspacing=0.5, borderpad=0.25,
                   framealpha=0.9, edgecolor='0.75', fancybox=False)

    norm = Normalize(vmin=h_vmin, vmax=h_vmax)
    sm = ScalarMappable(cmap='YlGnBu', norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Terrain height (mm)', fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6)

    means = []

    def _fmt(v):
        return f'{v:.1f}' if err_unit == 'cm' else f'{v:.3f}'

    def _draw_error_panel(ax, logs, title):
        if not flip_style:
            all_t, all_err, max_t = [], [], 0
            for idx, log in enumerate(logs):
                clr = cmap_tab10(idx % 10)
                ax.plot(log['t'], log['error'], color=clr, alpha=0.30,
                        linewidth=0.5)
                all_t.append(log['t']); all_err.append(log['error'])
                max_t = max(max_t, log['t'][-1])
            t_common = np.linspace(0, max_t, 500)
            interp_err = np.array([np.interp(t_common, ta, ea)
                                   for ta, ea in zip(all_t, all_err)])
            mean_err = np.mean(interp_err, axis=0)
            std_err = np.std(interp_err, axis=0)
            overall_mean = float(np.mean(mean_err))
            means.append(overall_mean)
            ax.fill_between(t_common, np.maximum(mean_err - std_err, 0),
                            mean_err + std_err, color='gray', alpha=0.20)
            ax.plot(t_common, mean_err, 'k-', linewidth=1.0, alpha=0.85,
                    label=f'Mean = {_fmt(overall_mean)} {err_unit}')
        else:
            # two-class error view: stable runs vs flipped runs
            is_flip = lambda l: l.get('flip_pos') is not None
            groups = [('Mean (not flip)',
                       [l for l in logs if not is_flip(l)],
                       '#2a5fb4', '#9ab8e0'),
                      ('Mean (flip runs)',
                       [l for l in logs if is_flip(l)],
                       '#c03a2b', '#e4a79e')]
            max_t = max(l['t'][-1] for l in logs)
            t_common = np.linspace(0, max_t, 600)
            tier_means = []
            for gname, gl, dark, light in groups:
                if not gl:
                    continue
                for log in gl:
                    ax.plot(log['t'], log['error'], color=light, alpha=0.55,
                            linewidth=0.5)
                # nan-aware mean (flipped runs truncate at different times)
                interp = np.full((len(gl), len(t_common)), np.nan)
                for k, log in enumerate(gl):
                    m_ = t_common <= log['t'][-1]
                    interp[k, m_] = np.interp(t_common[m_], log['t'],
                                              log['error'])
                mean_err = np.nanmean(interp, axis=0)
                overall = float(np.nanmean(mean_err))
                tier_means.append(overall)
                ax.plot(t_common, mean_err, '-', color=dark, linewidth=1.2,
                        label=f'{gname} = {_fmt(overall)} {err_unit}')
            means.append(float(np.mean(tier_means)))
        ax.set_xlabel('Time (s)'); ax.set_ylabel(f'Error ({err_unit})')
        ax.legend(fontsize=6, loc='upper left', framealpha=0.9,
                  edgecolor='0.8', borderpad=0.3)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        if err_row_caption is None:
            panel_caption(ax, title)

    # The error row is dropped at column width, so its one irreplaceable
    # number — the tier mean — moves into that tier's caption. Compute it off
    # a scratch figure that is never saved (it also stays the reproduction
    # check value returned to the caller).
    if single_column:
        scratch = plt.figure()
        for logs, _, err_title in tiers:
            _draw_error_panel(scratch.add_subplot(111), logs, err_title)
            scratch.clf()
        plt.close(scratch)
        for ax, (_, traj_title, _), m in zip(top_axes, tiers, means):
            panel_caption(ax, rf'{traj_title} ($\bar{{e}}$ = {m:.2f} {err_unit})')
    else:
        bot_axes = []
        for i, (logs, _, err_title) in enumerate(tiers):
            ax = fig.add_subplot(gs_bot[0, i])
            _draw_error_panel(ax, logs, err_title)
            bot_axes.append(ax)

        if err_row_caption:
            # hung off the middle panel, which the gridspec centres on the
            # row, so the row caption keeps the same font and drop as every
            # panel one
            panel_caption(bot_axes[1], err_row_caption)

    if footnote:
        fig.text(0.08, 0.005, footnote, fontsize=6.5, style='italic',
                 color='0.25')

    out_dir = Path(__file__).parent / 'plots'
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / out_name
    fig.savefig(str(out_pdf), format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_pdf}")
    return out_pdf, means


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark: multi-start terrain comparison (IEEE figure)')
    parser.add_argument('--roughness', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--hfield-res', type=int, default=600)
    parser.add_argument('--hfield-size', type=float, default=100.0)
    parser.add_argument('--duration', type=float, default=None,
                        help='Trajectory duration (auto = line_length/speed + 5)')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--line-length', type=float, default=30.0)
    parser.add_argument('--line-angle', type=float, default=0.0)
    parser.add_argument('--pid-rate', type=float, default=50.0)
    parser.add_argument('--mu-roll', type=float, default=0.01)
    parser.add_argument('--multi-start', type=int, default=4)
    parser.add_argument('--multi-start-radius', type=float, default=0.5)
    parser.add_argument('--unidirectional', action='store_true', default=True)
    parser.add_argument('--yaw-compensate', action='store_true', default=True)
    # RL
    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('--act-dim', type=int, default=3)
    parser.add_argument('--alpha-max', type=float, default=0.15)
    parser.add_argument('--ablation', action='store_true',
                        help='Add PID-only tier → 3-tier figure '
                             '(No-control / PID-only / PID+RL), reviewer R4-C8')
    parser.add_argument('--baseline-yawcomp', action='store_true',
                        help='Keep yaw-compensated drive on the no-control '
                             'tier (unfair baseline; off by default)')
    args = parser.parse_args()

    # Auto duration
    if args.duration is None:
        args.duration = args.line_length / max(args.speed, 1e-3) + 5.0

    # Trajectory endpoint
    angle_rad = np.radians(args.line_angle)
    end_x = args.line_length * np.cos(angle_rad)
    end_y = args.line_length * np.sin(angle_rad)

    # Load RL policy
    policy_path = args.policy or str(
        Path(__file__).parent / 'rl' / 'checkpoints' / 'old1' / 'best.pt')
    ckpt_info = peek_checkpoint(policy_path)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
    agent.load(policy_path)
    print(f"Loaded RL policy: {policy_path} (obs_dim={obs_dim})")

    env = SphericalRobotEnv(
        mu_roll=args.mu_roll, force_max=5, pid_rate=args.pid_rate,
        alpha_max=args.alpha_max, warmup_steps=0, max_episode_steps=999999,
        randomize=False, act_dim=args.act_dim,
        unidirectional=args.unidirectional, no_joint_obs=no_joint,
    )
    env.global_step = 999999
    env.controller.yaw_compensate = args.yaw_compensate
    controller = env.controller

    # Terrain
    terrain = RoughTerrain(
        roughness=args.roughness, seed=args.seed,
        nrow=args.hfield_res, ncol=args.hfield_res,
        size_x=args.hfield_size, size_y=args.hfield_size,
    )

    n_ms = args.multi_start
    radius = args.multi_start_radius
    offsets = np.linspace(-radius, radius, n_ms)

    print("=" * 60)
    print("Benchmark: Terrain Multi-Start Comparison")
    print(f"  roughness={args.roughness}  runs={n_ms}  "
          f"speed={args.speed}  length={args.line_length}m  "
          f"duration={args.duration:.1f}s")
    print(f"  Y offsets: {offsets}")
    print("=" * 60)

    baseline_logs = []
    pid_logs = []
    rl_logs = []

    for oi, y_off in enumerate(offsets):
        print(f"\n  Run {oi+1}/{n_ms}: y_offset={y_off:+.3f}m")

        # Baseline (no attitude control). For a FAIR no-control baseline the
        # drive must NOT be yaw-compensated either — otherwise the drive still
        # uses IMU yaw feedback and the robot tracks straight without any
        # attitude control (see audit §terrain-repro). Disable yaw_compensate
        # for this tier only unless --baseline-yawcomp is given.
        bl_traj = LineTrajectory(start=(0, 0), end=(end_x, end_y),
                                 speed=args.speed)
        bl_tracker = TrajectoryTracker(bl_traj, dt=1.0 / args.pid_rate)
        controller.reset()
        _saved_yc = controller.yaw_compensate
        if not args.baseline_yawcomp:
            controller.yaw_compensate = False
        bl_log = run_baseline_traj(
            controller, args.duration, terrain, bl_tracker,
            mu_roll=args.mu_roll, force_max=5,
            start_offset=(0.0, y_off))
        controller.yaw_compensate = _saved_yc
        baseline_logs.append(bl_log)
        err = bl_log['error']
        n_q = len(err) // 4
        print(f"    baseline: mean_err={np.mean(err[n_q:]):.3f}m  "
              f"max_err={np.max(err[n_q:]):.3f}m")

        # PID-only tier (attitude control ON, no RL residual) — ablation
        if args.ablation:
            pid_traj = LineTrajectory(start=(0, 0), end=(end_x, end_y),
                                      speed=args.speed)
            pid_tracker = TrajectoryTracker(pid_traj, dt=1.0 / args.pid_rate)
            controller.reset()
            pid_log = run_baseline_traj(
                controller, args.duration, terrain, pid_tracker,
                mu_roll=args.mu_roll, force_max=5,
                start_offset=(0.0, y_off), attitude_enabled=True)
            pid_logs.append(pid_log)
            err = pid_log['error']
            n_q = len(err) // 4
            print(f"    PID-only: mean_err={np.mean(err[n_q:]):.3f}m  "
                  f"max_err={np.max(err[n_q:]):.3f}m")

        # PID + RL
        rl_traj = LineTrajectory(start=(0, 0), end=(end_x, end_y),
                                 speed=args.speed)
        rl_tracker = TrajectoryTracker(rl_traj, dt=1.0 / args.pid_rate)
        controller.reset()
        rl_log = run_rl_traj(
            env, agent, controller, args.duration, terrain, rl_tracker,
            start_offset=(0.0, y_off))
        rl_logs.append(rl_log)
        err = rl_log['error']
        n_q = len(err) // 4
        print(f"    PID+RL:   mean_err={np.mean(err[n_q:]):.3f}m  "
              f"max_err={np.max(err[n_q:]):.3f}m")

    # Summary
    print(f"\n{'='*60}")
    print("Summary — Tracking Error (after transient)")
    print(f"{'='*60}")
    summary_tiers = [("No attitude", baseline_logs)]
    if args.ablation:
        summary_tiers.append(("PID-only", pid_logs))
    summary_tiers.append(("PID+RL", rl_logs))
    summary_rows = []
    for label, logs in summary_tiers:
        all_err = []
        for log in logs:
            err = log['error']
            n_q = len(err) // 4
            all_err.extend(err[n_q:])
        all_err = np.array(all_err)
        row = (label, float(np.mean(all_err)), float(np.std(all_err)),
               float(np.max(all_err)))
        summary_rows.append(row)
        print(f"  {label:14s}: mean={row[1]:.3f}m  "
              f"std={row[2]:.3f}m  max={row[3]:.3f}m")

    # Plot
    print("\nGenerating IEEE figure...")
    if args.ablation:
        tiers = [
            (baseline_logs, '(a) No attitude control',
             '(d) Tracking error — No attitude control'),
            (pid_logs, '(b) PID attitude control (no RL)',
             '(e) Tracking error — PID only'),
            (rl_logs, '(c) PID + Residual RL agent',
             '(f) Tracking error — PID + RL agent'),
        ]
        out_pdf, means = plot_comparison_ablation(tiers, terrain, args)

        # Quantitative CSV for the paper table (data/)
        data_dir = Path(__file__).resolve().parents[2] / 'data'
        data_dir.mkdir(exist_ok=True)
        csv_path = data_dir / 'terrain_ablation_tracking_error.csv'
        with open(csv_path, 'w') as f:
            f.write("condition,mean_err_m,std_err_m,max_err_m,"
                    "mean_of_run_means_m,roughness,n_runs,speed_mps,"
                    "line_length_m,seed\n")
            for (label, mean, std, mx), fig_mean in zip(summary_rows, means):
                f.write(f"{label},{mean:.4f},{std:.4f},{mx:.4f},"
                        f"{fig_mean:.4f},{args.roughness},{n_ms},"
                        f"{args.speed},{args.line_length},{args.seed}\n")
        print(f"Saved quantitative table → {csv_path}")
    else:
        plot_comparison(baseline_logs, rl_logs, terrain, args)


if __name__ == "__main__":
    main()
