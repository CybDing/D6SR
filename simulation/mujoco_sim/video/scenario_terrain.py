#!/usr/bin/env python3
"""
Scenario 2: Terrain switching — rolling across multiple terrain zones.

The robot follows a multi-directional waypoint trajectory through zones
of increasing roughness.  Each zone has a distinct visual texture and
movement pattern:
  Zone 1 (flat/white):   gentle S-curve
  Zone 2 (grass/green):  sinusoidal weave
  Zone 3 (gravel/rough): straight push through bumps

Desired trajectory is drawn on the ground with sphere markers + connecting
capsule lines for clear visual reference.  Actual robot path is drawn as
green trail dots.

Usage:
    cd mujoco_sim
    mjpython -m video.scenario_terrain --camera overview --duration 80
    mjpython -m video.scenario_terrain --camera follow
    mjpython -m video.scenario_terrain --camera side
    mjpython -m video.scenario_terrain --camera topdown
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from control.trajectory import WaypointTrajectory, TrajectoryTracker
from terrain import MultiZoneTerrain
from rl.env import SphericalRobotEnv, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from video.recorder import VideoRecorder
from video.cameras import terrain_cameras, FollowCamera
from video.visuals import (inject_video_visuals, inject_zone_boundary_strips,
                           paint_terrain_zones, inject_ref_and_trail_sites,
                           RefTrajectoryDrawer, TrailDrawer,
                           make_terrain_height_fn)


# ── Zone definitions ─────────────────────────────────────────────────────
# (x_start, x_end, roughness)
ZONES = [
    (-5,  8,  0.0),     # smooth flat
    (8,   20, 0.05),    # mild rough (grass)
    (20,  38, 0.15),    # rough (gravel)
]

# (x_start, x_end, (r, g, b), style) for texture painting
ZONE_COLORS = [
    (-5,  8,  (0.90, 0.89, 0.86), 'flat'),      # light white/grey — smooth
    (8,   20, (0.45, 0.68, 0.35), 'grass'),      # green — grass
    (20,  38, (0.58, 0.48, 0.38), 'gravel'),     # brown — gravel
]

# mu_roll per zone
ZONE_MU_ROLL = [
    (-5,  8,  0.01),
    (8,   20, 0.03),
    (20,  38, 0.06),
]

ZONE_BOUNDARIES = [8.0, 20.0]


def _set_shell_appearance(root, rgba='0.25 0.25 0.28 0.1'):
    """Override outer shell colour for better contrast against sky/terrain.

    Dark grey stands out against the blue sky and beige/green ground,
    resembling a real carbon-fibre shell.  alpha=0.25 with ~5 overlapping
    mesh layers gives ~76% effective opacity — visible structure but still
    lets internal gimbal show through.
    """
    for body in root.findall('.//body[@name="Link3"]'):
        for geom in body.findall('geom'):
            if geom.get('rgba'):
                geom.set('rgba', rgba)


def _get_mu_roll_for_x(x):
    """Return rolling friction coefficient based on robot X position."""
    for x_start, x_end, mu in ZONE_MU_ROLL:
        if x_start <= x < x_end:
            return mu
    return 0.03


def _build_waypoints(speed):
    """Build multi-directional waypoint path through the three zones.

    Zone 1 (flat, x=0..8):    gentle S-curve for graceful start
    Zone 2 (grass, x=8..20):  sinusoidal weave (side-to-side)
    Zone 3 (gravel, x=20..38): gentle curve then straight finish
    """
    pts = []

    # Start
    pts.append((0.0, 0.0))

    # Zone 1: S-curve
    pts.append((2.5,  1.2))
    pts.append((5.0, -0.5))
    pts.append((7.5,  0.8))

    # Zone 2: sinusoidal weave through grass
    n_weave = 6
    x_start, x_end = 8.5, 19.5
    amplitude = 2.5
    for i in range(n_weave + 1):
        frac = i / n_weave
        x = x_start + frac * (x_end - x_start)
        y = amplitude * np.sin(2 * np.pi * frac * 1.5)
        pts.append((x, y))

    # Zone 3: gentle curve then straight
    pts.append((22.0, -1.5))
    pts.append((26.0, -2.0))
    pts.append((30.0, -1.0))
    pts.append((35.0,  0.0))
    pts.append((38.0,  0.0))

    return pts


def run(args):
    speed = args.speed
    use_rl = bool(getattr(args, 'use_rl', 1))
    policy_path = getattr(args, 'policy', None)
    alpha_max = getattr(args, 'alpha_max', 0.15)
    act_dim = getattr(args, 'act_dim', 3)
    unidirectional = getattr(args, 'unidirectional', 2)

    # ── Waypoint trajectory ───────────────────────────────────────────────
    waypoints = _build_waypoints(speed)
    traj = WaypointTrajectory(waypoints=waypoints, speed=speed)
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=max(0.8, speed * 1.5))
    terrain_size_x = max(args.hfield_size, max(x for x, _ in waypoints) + 150.0)
    terrain_size_y = max(args.hfield_size, max(abs(y) for _, y in waypoints) + 150.0)

    # ── RL residual (optional, enabled by default) ───────────────────────
    agent = None
    env = None
    if use_rl:
        if policy_path is None:
            policy_path = str(Path(__file__).resolve().parent.parent /
                              'rl' / 'checkpoints' / 'old1' / 'best.pt')
        ckpt_info = peek_checkpoint(policy_path)
        obs_dim = ckpt_info['obs_dim']
        no_joint = (obs_dim == OBS_DIM_NO_JOINT)
        agent = SAC(obs_dim=obs_dim, act_dim=act_dim)
        agent.load(policy_path)
        print(f"Loaded RL policy: {policy_path} (obs_dim={obs_dim})")

        env = SphericalRobotEnv(
            mu_roll=args.mu_roll, pid_rate=args.pid_rate,
            alpha_max=alpha_max, warmup_steps=0,
            max_episode_steps=999999, randomize=False,
            act_dim=act_dim, unidirectional=unidirectional,
            no_joint_obs=no_joint,
        )
        env.global_step = 999999  # full alpha

    # ── Terrain ──────────────────────────────────────────────────────────
    terrain = MultiZoneTerrain(
        zones=ZONES, seed=42,
        nrow=args.hfield_res, ncol=args.hfield_res,
        size_x=terrain_size_x, size_y=terrain_size_y,
    )

    # ── Cameras + visuals ────────────────────────────────────────────────
    cameras = terrain_cameras(zone_boundary_x=ZONE_BOUNDARIES[1],
                              target_body='base_link')

    n_ref = min(800, int(traj.duration() / 0.12))
    n_trail = min(600, int(args.duration / 0.3))
    injectors = [
        partial(inject_video_visuals, ground_checker=False, fog_extent=100),
        partial(inject_zone_boundary_strips, boundaries=ZONE_BOUNDARIES),
        partial(inject_ref_and_trail_sites,
                n_ref=n_ref, n_trail=n_trail,
                ref_size=0.025, ref_rgba='0.95 0.05 0.15 0.85',
                trail_size=0.014, trail_rgba='0.05 0.55 0.95 0.90'),
        _set_shell_appearance,
    ]

    # ── Simulation ───────────────────────────────────────────────────────
    sim = RobotSimulation(
        mu_roll=args.mu_roll, force_max=5,
        terrain=terrain,
        cameras=cameras, xml_injectors=injectors,
    )
    sim.reset()

    # Paint zone colors into the heightfield texture
    paint_terrain_zones(sim.model, ZONE_COLORS,
                        nrow=args.hfield_res, ncol=args.hfield_res,
                        size_x=terrain_size_x)

    # Terrain height function for dynamic ref trajectory placement
    terrain_height_fn = make_terrain_height_fn(
        sim.model, size_x=terrain_size_x, size_y=terrain_size_y)

    # ── Controller + trajectory ──────────────────────────────────────────
    controller = BaselineController(pid_rate=args.pid_rate)
    if env is not None:
        env.sim = sim
        env.shell_id = sim._shell_body_id
        env.dt_physics = sim.dt
        env.steps_per_pid = max(1, int(round(1.0 / args.pid_rate / sim.dt)))
        env.controller = controller
    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = True
    if env is not None:
        env.target_yaw = np.radians(imu['euler'][2])

    # ── Settle ───────────────────────────────────────────────────────────
    print(f"Settling {args.settle}s ...")
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    # ── Recorder ─────────────────────────────────────────────────────────
    out_path = VideoRecorder.make_path('terrain', args.camera,
                                        base_dir=str(Path(__file__).resolve().parent.parent / 'videos'))
    rec = VideoRecorder(sim.model, sim.data, fps=args.fps)
    rec.start(out_path)
    spf = rec.steps_per_frame

    follow_cam = FollowCamera(sim.model, sim.data, cam_name='follow')
    topdown_cam = FollowCamera(sim.model, sim.data, cam_name='topdown',
                                offset=np.array([0.0, 0.0, 10.0]),
                                smooth_alpha=0.04, lock_height=True)
    ref_drawer = RefTrajectoryDrawer(
        sim.model, traj, n_markers=n_ref, ground_z=-0.46,
        height_fn=terrain_height_fn, spacing_m=0.12)
    trail = TrailDrawer(sim.model, sim.data, n_trail=n_trail)
    trail_interval = max(1, int(0.3 / sim.dt))

    # ── Main loop ────────────────────────────────────────────────────────
    shell_id = sim._shell_body_id
    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / args.pid_rate / dt)))
    total = int(args.duration / dt)
    traj_start = sim.time
    ctrl = np.zeros(6)

    mode = "pid+rl" if agent is not None else "pid"
    print(f"Recording terrain scenario ({args.duration}s, camera={args.camera}, "
          f"speed={speed:.2f} m/s, {len(waypoints)} waypoints, mode={mode}) ...")
    for i in range(total):
        # Dynamic mu_roll based on robot position
        ball_x = sim.data.xpos[shell_id][0]
        new_mu = _get_mu_roll_for_x(ball_x)
        if abs(sim.mu_roll - new_mu) > 1e-6:
            sim.mu_roll = new_mu

        # PID update
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)
            pid_ctrl = controller.update(sim.get_imu(), ball_vel)
            if agent is not None:
                obs = env._get_obs()
                action = agent.select_action(obs, deterministic=True)
                action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
                residual = env.alpha * action
                if unidirectional and act_dim == 3:
                    pre = controller.last_pre_alloc
                    ctrl = np.clip(
                        mix_motors(pre[0] + residual[0], pre[1] + residual[1],
                                   pre[2] + residual[2], pre[3], pre[4],
                                   unidirectional=True),
                        -1.0, 1.0)
                else:
                    residual_6 = env._mix_residual(residual)
                    ctrl = np.clip(pid_ctrl + residual_6, -1.0, 1.0)
            else:
                ctrl = pid_ctrl
            sim.set_ctrl(ctrl)

        sim.step()

        # Dynamic markers
        if i % trail_interval == 0:
            traj_t = sim.time - traj_start
            ref_drawer.update(traj_t, lookahead=2)
            trail.stamp()

        # Capture
        if i % spf == 0:
            follow_cam.update()
            topdown_cam.update()
            rec.capture(args.camera)

        # Progress
        if i % int(1.0 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            traj_t = sim.time - traj_start
            ball_pos = sim.data.xpos[shell_id]
            zone = "flat" if ball_x < 8 else ("grass" if ball_x < 20 else "gravel")
            print(f"\r  t={traj_t:5.1f}s  xy=({ball_x:+.2f},{ball_pos[1]:+.2f})  "
                  f"zone={zone:6s}  "
                  f"mu={sim.mu_roll:.3f}  R={r:+5.1f} P={p:+5.1f}",
                  end="", flush=True)

    rec.stop()
    rec.close()
    print(f"\nDone: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Record terrain switching video")
    p.add_argument('--camera', default='overview',
                   choices=['overview', 'follow', 'side', 'topdown'])
    p.add_argument('--duration', type=float, default=100.0)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--speed', type=float, default=0.7)
    p.add_argument('--pid-rate', type=float, default=50.0)
    p.add_argument('--mu-roll', type=float, default=0.01)
    p.add_argument('--hfield-res', type=int, default=800)
    p.add_argument('--hfield-size', type=float, default=50.0)
    p.add_argument('--use-rl', type=int, default=1)
    p.add_argument('--policy', type=str, default=None)
    p.add_argument('--alpha-max', type=float, default=0.15)
    p.add_argument('--act-dim', type=int, default=3)
    p.add_argument('--unidirectional', type=int, default=2)
    return p.parse_args()


if __name__ == '__main__':
    args = main()
    run(args)
