#!/usr/bin/env python3
"""
Scenario: Baseline PID straight-line driving — V0 vs V1 comparison.

Demonstrates:
  V0 (shell-fixed): motors rotate with the shell during rolling — no
     internal stabilisation, orientation couples to ground contact.
  V1 (gimbal):      inner platform holds a fixed orientation while
     the outer shell rolls freely — decoupled control.

Both robots drive a straight line with PID-only control on flat ground.
Use --model to select which robot to record.

Usage:
    cd mujoco_sim
    mjpython -m video.scenario_baseline --model v1 --camera follow
    mjpython -m video.scenario_baseline --model v0 --camera follow
    mjpython -m video.scenario_baseline --model v1 --camera overview
    mjpython -m video.scenario_baseline --model v0 --camera topdown
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from control.trajectory import WaypointTrajectory, TrajectoryTracker
from video.recorder import VideoRecorder
from video.cameras import default_cameras, FollowCamera
from video.visuals import (inject_video_visuals, inject_ref_and_trail_sites,
                           RefTrajectoryDrawer, TrailDrawer)


SHELL_RGBA = '0.25 0.25 0.28 0.1'

MODEL_PATHS = {
    'v0': 'models/robot_v0_mujoco.xml',
    'v1': 'models/robot_v1_mujoco.xml',
}


def _set_shell_appearance(root):
    """Dark grey shell — works for both V0 (base_link) and V1 (Link3)."""
    # V1: shell mesh is on Link3
    if root.find('.//body[@name="Link3"]') is not None:
        body_name = 'Link3'
    else:
        # V0: single body, shell mesh is on base_link
        body_name = 'base_link'
    for body in root.findall(f'.//body[@name="{body_name}"]'):
        for geom in body.findall('geom'):
            if geom.get('type', 'mesh') == 'mesh' and geom.get('rgba'):
                geom.set('rgba', SHELL_RGBA)


def run(args):
    speed = args.speed
    model_key = args.model
    is_v0 = (model_key == 'v0')
    mjcf = str(Path(__file__).resolve().parent.parent / MODEL_PATHS[model_key])

    # ── Straight-line trajectory ──────────────────────────────────────────
    distance = speed * args.duration * 0.85  # leave margin at end
    waypoints = [(0.0, 0.0), (distance, 0.0)]
    traj = WaypointTrajectory(waypoints=waypoints, speed=speed)
    tracker = TrajectoryTracker(traj, dt=1.0 / args.pid_rate,
                                max_speed=max(0.8, speed * 1.5))

    # ── Cameras ───────────────────────────────────────────────────────────
    cameras = default_cameras(target_body='base_link')
    mid_x = distance / 2
    cameras['overview']['pos'] = f'{mid_x:.0f} -5 4'
    cameras['overview']['fovy'] = '55'
    cameras['topdown']['pos'] = f'{mid_x:.0f} 0 8'
    cameras['topdown']['fovy'] = '55'

    # ── Visuals ───────────────────────────────────────────────────────────
    n_ref = min(400, int(traj.duration() / 0.15))
    n_trail = min(300, int(args.duration / 0.3))
    injectors = [
        partial(inject_video_visuals, white_bg=True),
        partial(inject_ref_and_trail_sites,
                n_ref=n_ref, n_trail=n_trail,
                ref_size=0.022, ref_rgba='0.95 0.05 0.15 0.85',
                trail_size=0.014, trail_rgba='0.05 0.55 0.95 0.90'),
        _set_shell_appearance,
    ]

    # ── Simulation ────────────────────────────────────────────────────────
    sim = RobotSimulation(
        mjcf_path=mjcf,
        mu_roll=args.mu_roll, force_max=5,
        cameras=cameras, xml_injectors=injectors,
    )
    sim.reset()

    controller = BaselineController(
        pid_rate=args.pid_rate,
        v0_mode=is_v0,
        motor_radius=sim._motor_radius,
        force_scale=2.0,
    )
    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = True

    # ── Settle ────────────────────────────────────────────────────────────
    print(f"Settling {args.settle}s ...")
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    # ── Recorder ──────────────────────────────────────────────────────────
    tag = f'baseline_{model_key}'
    out_path = VideoRecorder.make_path(
        tag, args.camera,
        base_dir=str(Path(__file__).resolve().parent.parent / 'videos'))
    rec = VideoRecorder(sim.model, sim.data, fps=args.fps)
    rec.start(out_path)
    spf = rec.steps_per_frame

    follow_cam = FollowCamera(sim.model, sim.data, cam_name='follow')
    topdown_cam = FollowCamera(sim.model, sim.data, cam_name='topdown',
                                offset=np.array([0.0, 0.0, 6.0]),
                                smooth_alpha=0.04, lock_height=True)

    ref_drawer = RefTrajectoryDrawer(
        sim.model, traj, n_markers=n_ref, ground_z=-0.46, spacing_m=0.15)
    trail = TrailDrawer(sim.model, sim.data, n_trail=n_trail)
    trail_interval = max(1, int(0.3 / sim.dt))

    # ── Main loop ─────────────────────────────────────────────────────────
    shell_id = sim._shell_body_id
    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / args.pid_rate / dt)))
    total = int(args.duration / dt)
    traj_start = sim.time

    label = 'V0 (shell-fixed)' if is_v0 else 'V1 (gimbal)'
    print(f"Recording {label} baseline ({args.duration}s, camera={args.camera}, "
          f"speed={speed:.2f} m/s) ...")
    for i in range(total):
        if i % steps_per_pid == 0:
            ball_pos = sim.data.xpos[shell_id][:2].copy()
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            traj_t = sim.time - traj_start

            vx, vy, ref_x, ref_y = tracker.compute(
                traj_t, ball_pos[0], ball_pos[1])
            controller.set_drive(vx, vy)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)

        sim.step()

        # Dynamic markers
        if i % trail_interval == 0:
            traj_t = sim.time - traj_start
            ref_drawer.update(traj_t, lookahead=1.5)
            trail.stamp()

        # Capture
        if i % spf == 0:
            follow_cam.update()
            topdown_cam.update()
            rec.capture(args.camera)

        # Progress
        if i % int(1.0 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            ball_pos = sim.data.xpos[shell_id]
            print(f"\r  t={sim.time - traj_start:5.1f}s  "
                  f"xy=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f})  "
                  f"R={r:+5.1f} P={p:+5.1f}",
                  end="", flush=True)

    rec.stop()
    rec.close()
    print(f"\nDone: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Record baseline straight-line video")
    p.add_argument('--model', default='v1', choices=['v0', 'v1'],
                   help='Robot model: v0 (shell-fixed) or v1 (gimbal)')
    p.add_argument('--camera', default='follow',
                   choices=['overview', 'follow', 'side', 'topdown'])
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--speed', type=float, default=0.6)
    p.add_argument('--pid-rate', type=float, default=50.0)
    p.add_argument('--mu-roll', type=float, default=0.01)
    return p.parse_args()


if __name__ == '__main__':
    args = main()
    run(args)
