#!/usr/bin/env python3
"""
Scenario 1: Circle trajectory tracking with PID + RL residual.

Records the robot following a circular reference path.
- Red/orange dots: desired reference trajectory (revealed dynamically)
- Green dots: actual robot path (trail)

Usage:
    cd mujoco_sim
    mjpython -m video.scenario_trajectory --camera overview --duration 60
    mjpython -m video.scenario_trajectory --camera follow --radius 5 --speed 0.5
    mjpython -m video.scenario_trajectory --camera topdown
    mjpython -m video.scenario_trajectory --trajectory text --text IROS2026 --camera topdown
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from control.trajectory import (CircleTrajectory, TextTrajectory,
                                TrajectoryTracker)
from rl.env import SphericalRobotEnv, OBS_DIM_NO_JOINT
from rl.sac import SAC, peek_checkpoint
from video.recorder import VideoRecorder
from video.cameras import trajectory_cameras, FollowCamera
from video.visuals import (inject_video_visuals, inject_ref_and_trail_sites,
                           RefTrajectoryDrawer, TrailDrawer)


SHELL_RGBA = '0.25 0.25 0.28 0.1'


def _set_shell_appearance(root):
    """Dark grey shell — consistent with other video scenarios."""
    for body in root.findall('.//body[@name="Link3"]'):
        for geom in body.findall('geom'):
            if geom.get('rgba'):
                geom.set('rgba', SHELL_RGBA)


def run(args):
    # ── Trajectory ───────────────────────────────────────────────────────
    if args.trajectory == 'text':
        traj = TextTrajectory(args.text, char_width=args.char_width,
                              char_height=args.char_height, speed=args.speed)
        # Camera framing: cover the full text extent
        xs = [p[0] for p in traj.waypoints]
        ys = [p[1] for p in traj.waypoints]
        cam_radius = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6
        # Auto-set duration to cover the full text path + buffer
        traj_dur = traj.duration()
        if args.duration == 60.0:  # default not overridden
            args.duration = traj_dur + 5.0
            print(f"Text trajectory duration: {traj_dur:.1f}s "
                  f"(recording {args.duration:.1f}s)")
    else:
        traj = CircleTrajectory(radius=args.radius, speed=args.speed,
                                center=(0, 0))
        cam_radius = args.radius
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=args.max_speed)

    # ── Cameras + visuals ────────────────────────────────────────────────
    cameras = trajectory_cameras(radius=cam_radius, target_body='base_link')
    if args.trajectory == 'text':
        # Reposition overview/topdown to center over the text
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        # overview → fixed top-down covering the entire text
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        # Height from FOV: h = span / (2*tan(fov/2)), use larger span + margin
        fov_rad = np.radians(75) / 2
        need_h = max(span_x, span_y) * 1.15 / (2 * np.tan(fov_rad))
        cameras['overview']['pos'] = f'{cx:.1f} {cy:.1f} {need_h:.1f}'
        cameras['overview']['xyaxes'] = '1 0 0 0 1 0'  # look straight down
        cameras['overview']['fovy'] = '55'
        cameras['topdown']['pos'] = f'{cx:.1f} {cy:.1f} {cam_radius * 1.6 + 2:.1f}'
    if args.trajectory == 'text':
        # Text paths are long — estimate markers from actual path length
        path_len = sum(
            np.hypot(traj.waypoints[i+1][0] - traj.waypoints[i][0],
                     traj.waypoints[i+1][1] - traj.waypoints[i][1])
            for i in range(len(traj.waypoints) - 1))
        n_ref = int(path_len / 0.15) + 50
        # trail stamps by time (every 0.25s), not path length
        n_trail = int(args.duration / 0.25) + 100
    else:
        n_ref = min(800, int(args.duration / 0.15))
        n_trail = min(600, int(args.duration / 0.25))
    injectors = [
        partial(inject_video_visuals, white_bg=True),
        partial(inject_ref_and_trail_sites,
                n_ref=n_ref, n_trail=n_trail,
                ref_size=0.030, ref_rgba='0.92 0.35 0.35 0.65',
                trail_size=0.045, trail_rgba='0.10 0.25 0.72 0.95'),
        _set_shell_appearance,
    ]

    # ── RL agent ─────────────────────────────────────────────────────────
    policy_path = args.policy
    if policy_path is None:
        policy_path = str(Path(__file__).resolve().parent.parent /
                          'rl' / 'checkpoints' / 'old1' / 'best.pt')
    ckpt_info = peek_checkpoint(policy_path)
    obs_dim = ckpt_info['obs_dim']
    no_joint = (obs_dim == OBS_DIM_NO_JOINT)
    agent = SAC(obs_dim=obs_dim, act_dim=args.act_dim)
    agent.load(policy_path)
    print(f"Loaded RL policy: {policy_path} (obs_dim={obs_dim})")

    # ── Environment (for obs construction + residual mixing) ─────────────
    env = SphericalRobotEnv(
        mu_roll=args.mu_roll, pid_rate=args.pid_rate,
        alpha_max=args.alpha_max, warmup_steps=0,
        max_episode_steps=999999, randomize=False,
        act_dim=args.act_dim, unidirectional=args.unidirectional,
        no_joint_obs=no_joint,
    )
    env.global_step = 999999  # full alpha

    # Replace env's sim with our camera-enabled sim
    sim = RobotSimulation(
        mu_roll=args.mu_roll, force_max=5,
        cameras=cameras, xml_injectors=injectors,
    )
    sim.reset()
    env.sim = sim
    env.shell_id = sim._shell_body_id
    env.dt_physics = sim.dt
    env.steps_per_pid = max(1, int(round(1.0 / args.pid_rate / sim.dt)))
    controller = env.controller

    # ── Settle ───────────────────────────────────────────────────────────
    print(f"Settling {args.settle}s ...")
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = True
    env.target_yaw = np.radians(imu['euler'][2])

    # ── Recorder ─────────────────────────────────────────────────────────
    out_path = VideoRecorder.make_path('trajectory', args.camera,
                                        base_dir=str(Path(__file__).resolve().parent.parent / 'videos'))
    rec = VideoRecorder(sim.model, sim.data, fps=args.fps)
    rec.start(out_path)
    spf = rec.steps_per_frame

    follow_cam = FollowCamera(sim.model, sim.data, cam_name='follow')
    topdown_cam = FollowCamera(sim.model, sim.data, cam_name='topdown',
                                offset=np.array([0.0, 0.0, 3.0]),
                                smooth_alpha=0.04, lock_height=True)

    # ── Main loop ────────────────────────────────────────────────────────
    shell_id = sim._shell_body_id
    dt = sim.dt
    ref_drawer = RefTrajectoryDrawer(
        sim.model, traj, n_markers=n_ref, ground_z=-0.46, spacing_m=0.15)
    trail = TrailDrawer(sim.model, sim.data, n_trail=n_trail)
    trail_interval = max(1, int(0.25 / dt))
    steps_per_pid = max(1, int(round(1.0 / args.pid_rate / dt)))
    total = int(args.duration / dt)
    traj_start = sim.time
    ctrl = np.zeros(6)

    print(f"Recording trajectory scenario ({args.duration}s, camera={args.camera}) ...")
    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)

            # PID output
            pid_ctrl = controller.update(sim.get_imu(), ball_vel)

            # RL residual
            obs = env._get_obs()
            action = agent.select_action(obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            residual = env.alpha * action

            if args.unidirectional and args.act_dim == 3:
                pre = controller.last_pre_alloc
                ctrl = np.clip(
                    mix_motors(pre[0] + residual[0], pre[1] + residual[1],
                               pre[2] + residual[2], pre[3], pre[4],
                               unidirectional=True),
                    -1.0, 1.0)
            else:
                residual_6 = env._mix_residual(residual)
                ctrl = np.clip(pid_ctrl + residual_6, -1.0, 1.0)

            sim.set_ctrl(ctrl)

        sim.step()

        # Dynamic markers
        if i % trail_interval == 0:
            traj_t = sim.time - traj_start
            ref_drawer.update(traj_t, lookahead=1.2)
            trail.stamp()

        # Capture
        if i % spf == 0:
            follow_cam.update()
            topdown_cam.update()
            rec.capture(args.camera)

        # Progress
        if i % int(2.0 / dt) == 0:
            traj_t = sim.time - traj_start
            ball_pos = sim.data.xpos[shell_id][:2]
            ref_x, ref_y = traj.position(traj_t)
            err = np.hypot(ball_pos[0] - ref_x, ball_pos[1] - ref_y)
            print(f"\r  t={traj_t:5.1f}s  err={err:.3f}m  "
                  f"pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})",
                  end="", flush=True)

    rec.stop()
    rec.close()
    print(f"\nDone: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Record trajectory tracking video")
    p.add_argument('--camera', default='overview',
                   choices=['overview', 'follow', 'side', 'topdown'])
    p.add_argument('--duration', type=float, default=60.0)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--trajectory', default='circle',
                   choices=['circle', 'text'])
    p.add_argument('--text', type=str, default='IROS2026',
                   help='Text string for text trajectory')
    p.add_argument('--char-width', type=float, default=2.4)
    p.add_argument('--char-height', type=float, default=6.0)
    p.add_argument('--radius', type=float, default=10.0)
    p.add_argument('--speed', type=float, default=0.8)
    p.add_argument('--max-speed', type=float, default=0.8)
    p.add_argument('--pid-rate', type=float, default=50.0)
    p.add_argument('--mu-roll', type=float, default=0.01)
    p.add_argument('--alpha-max', type=float, default=0.15)
    p.add_argument('--act-dim', type=int, default=3)
    p.add_argument('--unidirectional', type=int, default=2)
    p.add_argument('--yaw-compensate', type=int, default=0)
    p.add_argument('--policy', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = main()
    run(args)
