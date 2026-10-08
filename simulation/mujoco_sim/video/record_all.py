#!/usr/bin/env python3
"""
Batch-record all video scenarios with multiple cameras.

Runs each scenario × camera combination sequentially.
Each recording is a separate simulation run (deterministic physics,
so the same scenario with different cameras produces identical motion).

Usage:
    cd mujoco_sim
    mjpython -m video.record_all                           # all scenarios, all cameras
    mjpython -m video.record_all --scenarios trajectory     # one scenario, all cameras
    mjpython -m video.record_all --cameras overview follow  # all scenarios, selected cameras
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCENARIO_CONFIGS = {
    'disturbance': {
        'cameras': ['overview', 'follow', 'topdown'],
        'defaults': {'duration': 30.0, 'settle': 2.0},
    },
    'trajectory': {
        'cameras': ['overview', 'follow', 'topdown'],
        'defaults': {'duration': 60.0, 'settle': 2.0, 'radius': 5.0, 'speed': 0.5},
    },
    'terrain': {
        'cameras': ['overview', 'follow', 'side', 'topdown'],
        'defaults': {'duration': 100.0, 'settle': 2.0, 'speed': 0.5},
    },
}


def run_scenario(scenario_name, camera, overrides=None):
    """Import and run a single scenario with given camera."""
    if overrides is None:
        overrides = {}

    defaults = SCENARIO_CONFIGS[scenario_name]['defaults'].copy()
    defaults.update(overrides)
    defaults['camera'] = camera
    defaults.setdefault('fps', 60)
    defaults.setdefault('pid_rate', 50.0)
    defaults.setdefault('mu_roll', 0.03)

    # Build args namespace
    args = argparse.Namespace(**defaults)

    if scenario_name == 'disturbance':
        from video.scenario_disturbance import run
        run(args)
    elif scenario_name == 'trajectory':
        args.__dict__.setdefault('max_speed', 0.8)
        args.__dict__.setdefault('alpha_max', 0.15)
        args.__dict__.setdefault('act_dim', 3)
        args.__dict__.setdefault('unidirectional', 2)
        args.__dict__.setdefault('yaw_compensate', 0)
        args.__dict__.setdefault('policy', None)
        from video.scenario_trajectory import run
        run(args)
    elif scenario_name == 'terrain':
        args.__dict__.setdefault('hfield_res', 600)
        args.__dict__.setdefault('hfield_size', 50.0)
        args.mu_roll = 0.01  # terrain scenario starts with low friction
        from video.scenario_terrain import run
        run(args)


def main():
    p = argparse.ArgumentParser(description="Batch-record all demo videos")
    p.add_argument('--scenarios', nargs='+',
                   default=list(SCENARIO_CONFIGS.keys()),
                   choices=list(SCENARIO_CONFIGS.keys()),
                   help="Which scenarios to record")
    p.add_argument('--cameras', nargs='+', default=None,
                   help="Override camera list (default: per-scenario defaults)")
    p.add_argument('--duration-scale', type=float, default=1.0,
                   help="Scale all durations (0.1 for quick test)")
    args = p.parse_args()

    total_start = time.time()
    count = 0

    for scenario in args.scenarios:
        cfg = SCENARIO_CONFIGS[scenario]
        cameras = args.cameras or cfg['cameras']
        cameras = [c for c in cameras if c in ['overview', 'follow', 'side', 'topdown']]

        for camera in cameras:
            print(f"\n{'='*60}")
            print(f"  {scenario} / {camera}")
            print(f"{'='*60}\n")

            overrides = {}
            if args.duration_scale != 1.0:
                overrides['duration'] = cfg['defaults']['duration'] * args.duration_scale

            t0 = time.time()
            try:
                run_scenario(scenario, camera, overrides)
            except Exception as e:
                print(f"\nERROR in {scenario}/{camera}: {e}")
                import traceback
                traceback.print_exc()
                continue
            count += 1
            print(f"  ({time.time() - t0:.1f}s elapsed)")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Done! {count} videos recorded in {total:.0f}s")
    print(f"  Output: videos/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
