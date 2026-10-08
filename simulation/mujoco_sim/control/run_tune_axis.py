#!/usr/bin/env python3
"""
Single-axis PID tuning tool.

Usage:
    mjpython run_tune_axis.py              # headless + plot
    mjpython run_tune_axis.py --viewer      # interactive viewer (no plot)
"""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation import RobotSimulation
from control.cascaded_pid import CascadedPID, CascadedPIDConfig


# ============= CONFIG =============
AXIS         = 'roll'       # 'roll' / 'pitch' / 'yaw'
TARGET_ANGLE = -40.0          # target angle (deg)
DURATION     = 30.0          # test duration (s)

# Outer loop (angle -> velocity target)
OUTER_KP   = 0.32 * 1.4
OUTER_KI   = 0.02 * 1.4
OUTER_KD   = 0.0 
OUTER_MAX  = 60.0            # velocity target limit (deg/s)
DEADBAND   = 0.0             # angle deadband (deg)

# Inner loop (velocity -> motor output)
INNER_KP   = 0.0008 * 5 * 1.4
INNER_KI   = 0.0000
INNER_KD   = 0.00005 * 5 * 1.4
INNER_MAX  = 0.8             # motor output limit

# Gravity compensation (feedforward)
FORCE_SCALING_FACTOR = 0.4   # match run_baseline.py
UNIDIRECTIONAL = 2           # 2 when enabled (doubles grav comp), 1 when disabled
GRAV_GAIN  = 0.1365 * 5 * FORCE_SCALING_FACTOR * UNIDIRECTIONAL * 0   # 0 = off
GRAV_SIGN  = +1
STABLE_ANGLE = 0.0           # equilibrium angle (deg)

# Slew limits
VEL_SLEW   = 30.0            # deg/s per PID step
OUT_SLEW   = 0.003           # ctrl per PID step

# Simulation
SETTLE     = 2.0             # pre-settle time (s)
PID_RATE   = 50.0            # PID frequency (Hz)
MU_ROLL    = 0.01            # rolling friction
UNIDIRECTIONAL_MODE = True   # True = only positive motor commands (one motor at a time)

# Initial condition (INIT_ANGLE takes priority over KICK)
INIT_ANGLE    = 0.0          # static tilt before PID (deg, 0 = use kick instead)
KICK_STRENGTH = 0.0           # motor ctrl value during kick (0 = no kick)
KICK_DURATION = 0.0           # kick duration (s)
# ===============================================


# Axis -> (euler_idx, gyro_idx, motor_a, motor_b)
AXIS_MAP = {
    'roll':  (0, 0, 2, 3),
    'pitch': (1, 1, 4, 5),
    'yaw':   (2, 2, 0, 1),
}


def build_config():
    """Convert CONFIG constants into a CascadedPIDConfig."""
    return CascadedPIDConfig(
        outer_kp=OUTER_KP, outer_ki=OUTER_KI, outer_kd=OUTER_KD,
        outer_max_output=OUTER_MAX, outer_deadband=DEADBAND,
        inner_kp=INNER_KP, inner_ki=INNER_KI, inner_kd=INNER_KD,
        inner_max_output=INNER_MAX,
        dt=1.0 / PID_RATE,
        velocity_slew_rate=VEL_SLEW, output_slew_rate=OUT_SLEW,
        gravity_comp_gain=GRAV_GAIN, stable_angle_deg=STABLE_ANGLE,
        grav_sign=GRAV_SIGN,
        use_angle_wrap=True,
    )


def apply_kick(sim, axis_info):
    """Apply a short motor burst to tilt the ball away from center."""
    if KICK_STRENGTH == 0 or KICK_DURATION <= 0:
        return
    _, _, mot_a, mot_b = axis_info
    steps = int(KICK_DURATION / sim.dt)
    ctrl = np.zeros(6)
    if UNIDIRECTIONAL_MODE:
        ctrl[mot_a] = max(+KICK_STRENGTH, 0)
        ctrl[mot_b] = max(-KICK_STRENGTH, 0)
    else:
        ctrl[mot_a] = +KICK_STRENGTH
        ctrl[mot_b] = -KICK_STRENGTH
    sim.set_ctrl(ctrl)
    for _ in range(steps):
        sim.step()
    # Turn off motors, coast briefly to let angular velocity decay
    sim.set_ctrl(np.zeros(6))
    coast = int(0.5 / sim.dt)
    for _ in range(coast):
        sim.step()
    imu = sim.get_imu()
    euler_idx = axis_info[0]
    print(f"  Kick done: {AXIS}={imu['euler'][euler_idx]:+.1f}° "
          f"(ctrl={KICK_STRENGTH}, {KICK_DURATION}s + 0.5s coast)")


def set_initial_tilt(sim, axis_info):
    """Place ball at a specific tilt angle with zero velocity (static IC).

    Sets the freejoint quaternion to rotate the entire assembly by INIT_ANGLE
    around the test axis.  All velocities are zeroed.  Since Link3 is a sphere,
    the contact with ground remains valid after rotation.
    """
    euler_idx = axis_info[0]
    angle_rad = np.radians(INIT_ANGLE)
    half = angle_rad / 2.0

    # Quaternion [w, x, y, z] for rotation around test axis
    c, s = np.cos(half), np.sin(half)
    if AXIS == 'roll':
        quat = [c, s, 0, 0]       # rotate around X
    elif AXIS == 'pitch':
        quat = [c, 0, s, 0]       # rotate around Y
    else:  # yaw
        quat = [c, 0, 0, s]       # rotate around Z

    # Apply: keep settled position, replace orientation, zero velocity
    sim.data.qpos[3:7] = quat
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    imu = sim.get_imu()
    print(f"  Initial tilt set: {AXIS}={imu['euler'][euler_idx]:+.1f}° "
          f"(requested {INIT_ANGLE:+.1f}°, zero velocity)")


def run_test(sim, pid, axis_info):
    """Headless main loop. Returns log dict with numpy arrays."""
    euler_idx, gyro_idx, mot_a, mot_b = axis_info
    dt = sim.dt
    steps_per_pid = max(1, round(1.0 / PID_RATE / dt))

    keys = ['t', 'angle',
            'outer_p', 'outer_i', 'outer_d', 'vel_target',
            'inner_p', 'inner_i', 'inner_d',
            'grav_comp', 'output']
    log = {k: [] for k in keys}

    total = int(DURATION / dt)
    ctrl = np.zeros(6)

    for i in range(total):
        if i % steps_per_pid == 0:
            imu = sim.get_imu()
            angle = imu['euler'][euler_idx]
            gyro_deg = np.degrees(imu['gyro'][gyro_idx])
            output = pid.step(angle, gyro_deg)

            ctrl[:] = 0
            if UNIDIRECTIONAL_MODE:
                # Only positive motor commands — one motor fires at a time
                ctrl[mot_a] = max(+output, 0)
                ctrl[mot_b] = max(-output, 0)
            else:
                ctrl[mot_a] = +output
                ctrl[mot_b] = -output
            sim.set_ctrl(ctrl)

            # Record at PID rate
            log['t'].append(sim.time)
            log['angle'].append(angle)
            for k in keys[2:]:
                log[k].append(pid.dbg[k])

        # Progress printout every 0.5s
        if i % max(1, int(0.5 / dt)) == 0:
            imu = sim.get_imu()
            a = imu['euler'][euler_idx]
            print(f"\r  t={sim.time:5.1f}s  {AXIS}={a:+7.2f}°  "
                  f"target={TARGET_ANGLE:+.1f}°  ctrl={ctrl[mot_a]:+.4f}",
                  end="", flush=True)

        sim.step()

    print()
    return {k: np.array(v) for k, v in log.items()}


def run_viewer(sim, pid, axis_info):
    """Interactive viewer mode — same control logic, no data logging."""
    import mujoco.viewer

    euler_idx, gyro_idx, mot_a, mot_b = axis_info
    dt = sim.dt
    steps_per_pid = max(1, round(1.0 / PID_RATE / dt))
    ctrl = np.zeros(6)
    step_i = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        print("Viewer launched. Close window to exit.")
        while viewer.is_running():
            t0 = time.perf_counter()

            if step_i % steps_per_pid == 0:
                imu = sim.get_imu()
                angle = imu['euler'][euler_idx]
                gyro_deg = np.degrees(imu['gyro'][gyro_idx])
                output = pid.step(angle, gyro_deg)

                ctrl[:] = 0
                if UNIDIRECTIONAL_MODE:
                    ctrl[mot_a] = max(+output, 0)
                    ctrl[mot_b] = max(-output, 0)
                else:
                    ctrl[mot_a] = +output
                    ctrl[mot_b] = -output
                sim.set_ctrl(ctrl)

            if step_i % max(1, int(0.5 / dt)) == 0:
                imu = sim.get_imu()
                a = imu['euler'][euler_idx]
                print(f"\r  t={sim.time:5.1f}s  {AXIS}={a:+7.2f}°  "
                      f"target={TARGET_ANGLE:+.1f}°  ctrl={ctrl[mot_a]:+.4f}",
                      end="", flush=True)

            sim.step()
            viewer.sync()
            step_i += 1
            sl = dt - (time.perf_counter() - t0)
            if sl > 0:
                time.sleep(sl)
    print()


def plot(log):
    """4-subplot diagnostic plot."""
    import matplotlib.pyplot as plt

    t = log['t']
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

    # Title with current config
    title = (f"{AXIS.upper()} target={TARGET_ANGLE}° | "
             f"outer: Kp={OUTER_KP} Ki={OUTER_KI} Kd={OUTER_KD} | "
             f"inner: Kp={INNER_KP} Ki={INNER_KI} Kd={INNER_KD} | "
             f"grav={GRAV_GAIN}")
    if INIT_ANGLE != 0:
        title += f" | init={INIT_ANGLE}°"
    elif KICK_STRENGTH > 0:
        title += f" | kick={KICK_STRENGTH}/{KICK_DURATION}s"
    fig.suptitle(title, fontsize=10, fontfamily='monospace')

    # --- Subplot 1: Angle tracking (unwrap to remove ±180° jumps) ---
    ax = axes[0]
    log['angle'] = np.degrees(np.unwrap(np.radians(log['angle'])))
    ax.plot(t, log['angle'], 'b-', lw=1.2, label='actual')
    ax.axhline(TARGET_ANGLE, color='r', ls='--', lw=1, label='target')
    ax.fill_between(t, TARGET_ANGLE - 2, TARGET_ANGLE + 2,
                    color='green', alpha=0.1, label='±2° band')
    ax.set_ylabel('Angle (°)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Outer loop ---
    ax = axes[1]
    ax.plot(t, log['outer_p'], label='outer_p', alpha=0.8)
    ax.plot(t, log['outer_i'], label='outer_i', alpha=0.8)
    ax.plot(t, log['outer_d'], label='outer_d', alpha=0.8)
    ax.plot(t, log['vel_target'], 'k-', lw=1.2, label='vel_target')
    ax.set_ylabel('Outer PID (°/s)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Inner loop + gravity comp ---
    ax = axes[2]
    ax.plot(t, log['inner_p'], label='inner_p', alpha=0.8)
    ax.plot(t, log['inner_i'], label='inner_i', alpha=0.8)
    ax.plot(t, log['inner_d'], label='inner_d', alpha=0.8)
    ax.plot(t, log['grav_comp'], 'm-', lw=1, label='grav_comp')
    ax.plot(t, log['output'], 'k-', lw=1.2, label='output')
    ax.set_ylabel('Inner + Grav')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: Motor ctrl ---
    ax = axes[3]
    output = log['output']
    if UNIDIRECTIONAL_MODE:
        mot_a_vals = np.maximum(output, 0)
        mot_b_vals = np.maximum(-output, 0)
    else:
        mot_a_vals = output
        mot_b_vals = -output
    ax.plot(t, mot_a_vals, 'b-', lw=1, label=f'M{AXIS_MAP[AXIS][2]} (+)')
    ax.plot(t, mot_b_vals, 'r-', lw=1, label=f'M{AXIS_MAP[AXIS][3]} (-)')
    ax.set_ylabel('Motor ctrl')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(__file__).parent / f'tune_{AXIS}.png'
    plt.savefig(str(out), dpi=150)
    print(f"Saved {out}")
    backend = plt.get_backend().lower()
    if 'agg' not in backend:
        try:
            plt.show()
        except Exception:
            pass


def main():
    viewer_mode = '--viewer' in sys.argv

    if AXIS not in AXIS_MAP:
        print(f"Unknown axis '{AXIS}'. Use: {list(AXIS_MAP.keys())}")
        sys.exit(1)

    print(f"=== Single-axis tune: {AXIS.upper()} ===")
    print(f"  target={TARGET_ANGLE}°  duration={DURATION}s  pid_rate={PID_RATE}Hz")
    print(f"  outer: Kp={OUTER_KP} Ki={OUTER_KI} Kd={OUTER_KD}  max={OUTER_MAX}")
    print(f"  inner: Kp={INNER_KP} Ki={INNER_KI} Kd={INNER_KD}  max={INNER_MAX}")
    print(f"  grav_gain={GRAV_GAIN}  grav_sign={GRAV_SIGN}  mu_roll={MU_ROLL}"
          f"  unidirectional={UNIDIRECTIONAL_MODE}")
    if INIT_ANGLE != 0:
        print(f"  init_angle={INIT_ANGLE}° (static IC, zero velocity)")
    else:
        print(f"  kick: strength={KICK_STRENGTH}  duration={KICK_DURATION}s")

    sim = RobotSimulation(mu_roll=MU_ROLL, force_max=5)
    sim.reset()

    cfg = build_config()
    pid = CascadedPID(cfg, AXIS.upper())

    # Settle
    print(f"\nSettling ({SETTLE}s)...")
    for _ in range(int(SETTLE / sim.dt)):
        sim.step()

    # Initial condition: static tilt or kick
    if INIT_ANGLE != 0:
        set_initial_tilt(sim, AXIS_MAP[AXIS])
    else:
        apply_kick(sim, AXIS_MAP[AXIS])

    pid.set_target(TARGET_ANGLE)

    if viewer_mode:
        run_viewer(sim, pid, AXIS_MAP[AXIS])
    else:
        log = run_test(sim, pid, AXIS_MAP[AXIS])

        # Steady-state stats (second half)
        a = log['angle']
        n = len(a) // 2
        ss = a[n:]
        err = ss - TARGET_ANGLE
        print(f"\nSteady-state (t>{log['t'][n]:.1f}s):")
        print(f"  mean={np.mean(ss):.2f}°  std={np.std(ss):.2f}°  "
              f"error={np.mean(err):+.2f}°  max_err={np.max(np.abs(err)):.2f}°")

        plot(log)


if __name__ == "__main__":
    main()
