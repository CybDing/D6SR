#!/usr/bin/env python3
"""
Scenario 3: Disturbance rejection — external force pushes on the shell.

The robot holds position (station-keeping) while scripted force impulses
push it sideways. PID attitude control recovers after each push.

Usage:
    cd mujoco_sim
    mjpython -m video.scenario_disturbance --camera overview --duration 30
    mjpython -m video.scenario_disturbance --camera follow
    mjpython -m video.scenario_disturbance --camera topdown
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel
from video.recorder import VideoRecorder
from video.cameras import disturbance_cameras, FollowCamera
from video.visuals import inject_video_visuals


# ── Scripted disturbance events ──────────────────────────────────────────

DEFAULT_DISTURBANCES = [
    # (start_time, duration, force_xyz)
    # Ball mass ~1.8kg. Need >>mg forces for dramatic visible displacement.
    (4.0,  0.5, np.array([25.0,   0.0,  0.0])),   # strong push +X
    (11.0, 0.5, np.array([0.0,  -25.0,  0.0])),   # strong push -Y
    (18.0, 0.6, np.array([18.0,  18.0,  8.0])),   # diagonal + upward lift
    (25.0, 0.4, np.array([-30.0,  0.0,  0.0])),   # very strong push -X
]


def run(args):
    cameras = disturbance_cameras(target_body='base_link')
    injectors = [inject_video_visuals]

    sim = RobotSimulation(
        mu_roll=args.mu_roll, force_max=5,
        cameras=cameras, xml_injectors=injectors,
    )
    sim.reset()

    # Controller: station-keeping (no drive)
    controller = BaselineController(pid_rate=args.pid_rate)
    imu = sim.get_imu()
    controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
    controller.enabled = True
    controller.set_drive(0, 0)

    # Settle
    print(f"Settling {args.settle}s ...")
    for _ in range(int(args.settle / sim.dt)):
        sim.step()

    # Recorder
    out_path = VideoRecorder.make_path('disturbance', args.camera,
                                        base_dir=str(Path(__file__).resolve().parent.parent / 'videos'))
    rec = VideoRecorder(sim.model, sim.data, fps=args.fps)
    rec.start(out_path)
    spf = rec.steps_per_frame

    # Follow camera
    follow_cam = FollowCamera(sim.model, sim.data, cam_name='follow')

    # Disturbance schedule
    disturbances = DEFAULT_DISTURBANCES
    shell_id = sim._shell_body_id
    dt = sim.dt
    steps_per_pid = max(1, int(round(1.0 / args.pid_rate / dt)))
    total = int(args.duration / dt)
    ctrl = np.zeros(6)
    t_start = sim.time

    print(f"Recording disturbance scenario ({args.duration}s, camera={args.camera}) ...")
    for i in range(total):
        t = sim.time - t_start

        # Apply / clear disturbance forces on shell.
        # Only write force [:3]; leave torque [3:] for rolling friction.
        active_force = None
        for t_on, dur, force in disturbances:
            if t_on <= t < t_on + dur:
                active_force = force
                break
        if active_force is not None:
            sim.data.xfrc_applied[shell_id, :3] = active_force
        else:
            sim.data.xfrc_applied[shell_id, :3] = 0

        # PID update
        if i % steps_per_pid == 0:
            ball_vel = get_ball_vel(sim.model, sim.data, shell_id)
            ctrl = controller.update(sim.get_imu(), ball_vel)
            sim.set_ctrl(ctrl)

        sim.step()

        # Capture frame
        if i % spf == 0:
            follow_cam.update()
            rec.capture(args.camera)

        # Progress
        if i % int(1.0 / dt) == 0:
            r, p, y = sim.get_imu()['euler']
            status = "PUSH" if active_force is not None else "    "
            print(f"\r  t={t:5.1f}s  R={r:+6.1f} P={p:+6.1f} Y={y:+6.1f}  {status}",
                  end="", flush=True)

    rec.stop()
    rec.close()
    print(f"\nDone: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Record disturbance rejection video")
    p.add_argument('--camera', default='overview',
                   choices=['overview', 'follow', 'side', 'topdown'])
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--pid-rate', type=float, default=50.0)
    p.add_argument('--mu-roll', type=float, default=0.03)
    return p.parse_args()


if __name__ == '__main__':
    args = main()
    run(args)
