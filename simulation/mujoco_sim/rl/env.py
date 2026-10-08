"""
Gym-like environment wrapper for residual RL on the spherical robot.

Wraps RobotSimulation + BaselineController. The RL policy outputs a structured
residual that is added to PID output after mixing to 6 motors.

Action modes:
  act_dim=3: attitude-only residual [roll_diff, pitch_diff, yaw_diff]
             PID handles velocity drive alone — RL only corrects attitude.
  act_dim=5: full residual [roll_diff, pitch_diff, yaw_diff, x_drive, y_drive]
             RL also adjusts forward/lateral common-mode thrust.

The SAC policy already outputs tanh-squashed actions in [-1, 1].
The env only scales by alpha — NO extra tanh (avoids double-squashing).

Data units:
  imu['euler']:  degrees  (from simulation.py _quat_to_euler)
  imu['gyro']:   rad/s    (raw MuJoCo gyro sensor)

Usage:
    env = SphericalRobotEnv(act_dim=3)  # attitude-only residual
    obs = env.reset()
    for _ in range(500):
        action = np.random.uniform(-1, 1, size=3)
        obs, reward, done, info = env.step(action)
        if done: break
"""

import sys
from collections import deque
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation import RobotSimulation
from control.run_baseline import BaselineController, get_ball_vel, mix_motors
from terrain import RoughTerrain

OBS_DIM_FULL = 15    # with joint angles (sin/cos of j1, j2)
OBS_DIM_NO_JOINT = 11  # without joint angles — IMU + gyro + vel only
OBS_DIM = OBS_DIM_FULL  # backward compatibility
GYRO_SCALE = 10.0    # rad/s normalization
V_MAX = 2.0          # m/s normalization
FLIP_THRESHOLD = 60  # degrees, early termination


def wrap_angle(rad):
    """Wrap angle to [-pi, pi]."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


class SphericalRobotEnv:
    def __init__(
        self,
        force_max=5,
        mu_roll=0.01,
        pid_rate=50,
        alpha_max=0.15,
        warmup_steps=50000,
        max_episode_steps=500,
        settle_seconds=0.5,
        randomize=True,
        act_dim=3,
        unidirectional=False,
        terrain_prob=0.5,
        roughness_range=(0.01, 0.04),
        terrain_switch_interval=1,
        no_joint_obs=False,
        shell_inertia=None,
        inertia_scales=None,
        inertia_overrides=None,
        mass_overrides=None,
        pos_overrides=None,
        dof_damping_scale=None,
        joint_frictionloss=None,
        joint_damping=None,
        attitude_gain_scale=1.0,
        attitude_gains=None,
        yaw_reward_weight=0.0,
        act_penalty=10.0,
        smooth_penalty=2.0,
        imu_delay_steps=0,
        imu_angle_noise=0.0,
        imu_gyro_noise=0.0,
        thrust_tau=None,
        frame_stack=1,
        terrain_kwargs=None,
    ):
        assert act_dim in (3, 5), f"act_dim must be 3 or 5, got {act_dim}"
        self.force_max = force_max
        self.mu_roll = mu_roll
        self.pid_rate = pid_rate
        self.alpha_max = alpha_max
        self.warmup_steps = warmup_steps
        self.max_episode_steps = max_episode_steps
        self.settle_seconds = settle_seconds
        self.randomize = randomize
        self.act_dim = act_dim
        self.unidirectional = unidirectional
        self.terrain_prob = terrain_prob
        self.roughness_range = roughness_range
        self.terrain_switch_interval = terrain_switch_interval
        self.no_joint_obs = no_joint_obs
        self.shell_inertia = shell_inertia
        # frame_stack > 1: observation = concat of the last N single frames.
        # Temporal context lets the policy estimate true state under IMU
        # noise/latency (realism pack), where a single frame is non-Markov.
        self.frame_stack = int(frame_stack)
        self._frames = deque(maxlen=self.frame_stack)
        self.yaw_reward_weight = yaw_reward_weight
        self.act_penalty = act_penalty
        self.smooth_penalty = smooth_penalty
        # Sensor/actuator realism (all default off = ideal legacy behavior):
        #   imu_delay_steps: control-loop latency in PID periods (20 ms each)
        #   imu_angle_noise: Gaussian std on euler angles fed to controller/obs (deg)
        #   imu_gyro_noise:  Gaussian std on gyro (rad/s)
        #   thrust_tau:      first-order ESC+prop thrust lag time constant (s)
        self.imu_delay_steps = int(imu_delay_steps)
        self.imu_angle_noise = imu_angle_noise
        self.imu_gyro_noise = imu_gyro_noise
        self.thrust_tau = thrust_tau
        self._imu_buf = deque(maxlen=self.imu_delay_steps + 1)
        self._applied_ctrl = np.zeros(6)
        self._ctrl_imu = None
        self.obs_dim = (OBS_DIM_NO_JOINT if no_joint_obs else OBS_DIM_FULL) \
            * self.frame_stack
        self._episode_count = 0
        self._current_terrain_label = 'flat'

        # Always include hfield in XML so model.hfield_data is available for in-place updates.
        # roughness_max sizes elevation_max to cover the full training range.
        _base_terrain = RoughTerrain(roughness=roughness_range[1],
                                     **(terrain_kwargs or {}))
        self.sim = RobotSimulation(mu_roll=mu_roll, force_max=force_max,
                                   terrain=_base_terrain,
                                   shell_inertia=shell_inertia,
                                   inertia_scales=inertia_scales,
                                   inertia_overrides=inertia_overrides,
                                   mass_overrides=mass_overrides,
                                   pos_overrides=pos_overrides,
                                   dof_damping_scale=dof_damping_scale,
                                   joint_frictionloss=joint_frictionloss,
                                   joint_damping=joint_damping)
        self._hfield_n = _base_terrain.nrow * _base_terrain.ncol
        self.controller = BaselineController(pid_rate=pid_rate,
                                             unidirectional=unidirectional,
                                             attitude_gain_scale=attitude_gain_scale,
                                             attitude_gains=attitude_gains)
        self.dt_physics = self.sim.dt
        self.steps_per_pid = max(1, int(round(1.0 / pid_rate / self.dt_physics)))
        self.shell_id = self.sim._shell_body_id
        self.pid_dt = 1.0 / pid_rate

        # State tracking
        self.global_step = 0
        self.episode_step = 0
        self.target_yaw = 0.0
        self.last_residual = np.zeros(act_dim)
        self.last_pid_ctrl = np.zeros(6)
        self.prev_residual = np.zeros(act_dim)  # for residual smoothness penalty

    @property
    def alpha(self):
        return self.alpha_max * min(1.0, self.global_step / max(1, self.warmup_steps))

    def _switch_terrain(self):
        """Randomly switch between flat and rough terrain in-place.

        Uses Terrain.populate() so the surface at the origin is re-centered to
        z = -0.5 and hfield elevation/geom offset stay consistent (writing raw
        heights here used to leave the robot embedded in / floating above the
        new terrain at episode start).
        """
        import mujoco
        if np.random.random() < self.terrain_prob:
            roughness = np.random.uniform(*self.roughness_range)
            seed = np.random.randint(0, 100000)
            nside = int(self._hfield_n ** 0.5)
            t = RoughTerrain(roughness=roughness, seed=seed, nrow=nside, ncol=nside)
            t.populate(self.sim.model)
            self._current_terrain_label = f'rough(r={roughness:.3f})'
        else:
            self.sim.model.hfield_data[:self._hfield_n] = 0.0
            gid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    'ground')
            if gid >= 0:
                self.sim.model.geom_pos[gid][2] = -0.5
            self._current_terrain_label = 'flat'

    def reset(self, target_vx=None, target_vy=None):
        """Reset simulation, settle with pure PID, return initial observation."""
        self._episode_count += 1
        if self._episode_count % self.terrain_switch_interval == 0:
            self._switch_terrain()
        self.sim.reset()
        self.controller.reset()
        self.episode_step = 0
        self.last_residual = np.zeros(self.act_dim)
        self.last_pid_ctrl = np.zeros(6)
        self.prev_residual = np.zeros(self.act_dim)

        # Domain randomization: perturb initial joint angles and rolling friction
        if self.randomize:
            state = self.sim.get_state()
            qpos = state['qpos'].copy()
            # Gimbal joints are at indices 7,8,9 (after freejoint 7-dof)
            qpos[7] += np.random.uniform(-0.3, 0.3)  # joint1 ±17°
            qpos[8] += np.random.uniform(-0.3, 0.3)  # joint2 ±17°
            qpos[9] += np.random.uniform(-0.3, 0.3)  # joint3 ±17°
            self.sim.reset(qpos=qpos)
            # Randomize rolling friction ±50%
            self.sim.mu_roll = self.mu_roll * np.random.uniform(0.5, 1.5)

        # Randomize target velocity if not specified
        if target_vx is None:
            target_vx = np.random.uniform(-1.0, 1.0)
        if target_vy is None:
            target_vy = np.random.uniform(-1.0, 1.0)

        # Settle with pure PID (no RL)
        settle_steps = int(self.settle_seconds / self.dt_physics)
        for _ in range(settle_steps):
            self.sim.step()

        imu = self.sim.get_imu()
        self.target_yaw = np.radians(imu['euler'][2])  # euler is degrees -> store as rad
        self.controller.set_target(roll=0, pitch=0, yaw=imu['euler'][2])
        self.controller.set_drive(vx=target_vx, vy=target_vy)
        self.controller.enabled = True

        self._imu_buf.clear()
        self._applied_ctrl = np.zeros(6)
        self._ctrl_imu = self._sense_imu()
        self._frames.clear()

        return self._get_obs()

    def _sense_imu(self):
        """IMU as the controller/policy sees it: noise + latency (realism pack).

        With the default parameters this returns the ideal IMU unchanged."""
        imu = self.sim.get_imu()
        if self.imu_angle_noise > 0 or self.imu_gyro_noise > 0:
            imu = dict(imu)
            imu['euler'] = imu['euler'] + np.random.normal(
                0.0, self.imu_angle_noise, 3)
            imu['gyro'] = imu['gyro'] + np.random.normal(
                0.0, self.imu_gyro_noise, 3)
        self._imu_buf.append(imu)
        return self._imu_buf[0]           # oldest = delayed by imu_delay_steps

    def _mix_residual(self, residual):
        """Mix structured residual to 6 motors.

        act_dim=3: [roll_diff, pitch_diff, yaw_diff]
        act_dim=5: [roll_diff, pitch_diff, yaw_diff, x_drive, y_drive]

        Motor layout: M0,M1=yaw+y_drive, M2,M3=roll, M4,M5=pitch+x_drive
        """
        res_roll = residual[0]
        res_pitch = residual[1]
        res_yaw = residual[2]
        res_x = residual[3] if self.act_dim == 5 else 0.0
        res_y = residual[4] if self.act_dim == 5 else 0.0
        return np.array([
            +res_yaw + res_y,    # M0
            -res_yaw + res_y,    # M1
            +res_roll,           # M2
            -res_roll,           # M3
            +res_pitch + res_x,  # M4
            -res_pitch + res_x,  # M5
        ])

    def step(self, action):
        """
        action: np.array of shape (act_dim,) in [-1, 1] (SAC already applies tanh)
        Returns: (obs, reward, done, info)
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Compute PID action (on the sensed = noisy/delayed IMU)
        imu = self._ctrl_imu = self._sense_imu()
        ball_vel = get_ball_vel(self.sim.model, self.sim.data, self.shell_id)
        pid_ctrl = self.controller.update(imu, ball_vel)
        self.last_pid_ctrl = pid_ctrl.copy()

        # Scale residual by alpha schedule (NO tanh — SAC output is already [-1,1])
        residual = self.alpha * action
        self.last_residual = residual.copy()

        if self.unidirectional and self.act_dim == 3:
            # Combine PID attitude + RL residual at pre-alloc level, then mix once
            pre = self.controller.last_pre_alloc  # [ro, po, yo, xd, yd]
            combined_ro = pre[0] + residual[0]
            combined_po = pre[1] + residual[1]
            combined_yo = pre[2] + residual[2]
            ctrl_total = np.clip(
                mix_motors(combined_ro, combined_po, combined_yo,
                           pre[3], pre[4], unidirectional=True),
                -1.0, 1.0)
        else:
            # Standard: PID mixed ctrl + separately mixed residual
            residual_6 = self._mix_residual(residual)
            ctrl_total = np.clip(pid_ctrl + residual_6, -1.0, 1.0)

        # First-order ESC+prop thrust lag (realism pack; tau=None -> ideal)
        if self.thrust_tau:
            beta = 1.0 - np.exp(-(1.0 / self.pid_rate) / self.thrust_tau)
            self._applied_ctrl = self._applied_ctrl + \
                beta * (ctrl_total - self._applied_ctrl)
            self.sim.set_ctrl(self._applied_ctrl)
        else:
            self.sim.set_ctrl(ctrl_total)

        # Step physics multiple times (PID rate < physics rate)
        for _ in range(self.steps_per_pid):
            self.sim.step()

        self.episode_step += 1
        self.global_step += 1

        obs = self._get_obs()
        reward = self._compute_reward(residual)
        done = self._check_done()

        self.prev_residual = residual.copy()

        info = {
            'pid_ctrl': pid_ctrl,
            'residual': residual,
            'alpha': self.alpha,
            'episode_step': self.episode_step,
        }
        return obs, reward, done, info

    def _get_obs(self):
        frame = self._get_obs_frame()
        if self.frame_stack == 1:
            return frame
        if not self._frames:
            for _ in range(self.frame_stack):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        return np.concatenate(self._frames)

    def _get_obs_frame(self):
        # Policy sees the same sensed (noisy/delayed) IMU as the PID
        imu = self._ctrl_imu if self._ctrl_imu is not None else self.sim.get_imu()
        euler_deg = imu['euler']   # degrees
        gyro = imu['gyro']         # rad/s

        roll_rad = np.radians(euler_deg[0])
        pitch_rad = np.radians(euler_deg[1])
        yaw_rad = np.radians(euler_deg[2])
        yaw_err = wrap_angle(self.target_yaw - yaw_rad)

        # Velocity error
        ball_vel = get_ball_vel(self.sim.model, self.sim.data, self.shell_id)
        vx_err = self.controller.target_vx - ball_vel[0]
        vy_err = self.controller.target_vy - ball_vel[1]

        # Base obs: attitude (sin/cos) + gyro + velocity error
        # These are all available from IMU + odometry on real robot
        obs_list = [
            np.sin(roll_rad), np.cos(roll_rad),
            np.sin(pitch_rad), np.cos(pitch_rad),
            np.sin(yaw_err), np.cos(yaw_err),
            gyro[0] / GYRO_SCALE, gyro[1] / GYRO_SCALE, gyro[2] / GYRO_SCALE,
        ]

        if not self.no_joint_obs:
            # Gimbal joint angles (radians from MuJoCo) — NOT available on real robot IMU
            j1_pos, _ = self.sim.get_joint_state('joint1')
            j2_pos, _ = self.sim.get_joint_state('joint2')
            obs_list.extend([
                np.sin(j1_pos), np.cos(j1_pos),
                np.sin(j2_pos), np.cos(j2_pos),
            ])

        obs_list.extend([
            np.clip(vx_err / V_MAX, -1.0, 1.0),
            np.clip(vy_err / V_MAX, -1.0, 1.0),
        ])

        return np.array(obs_list, dtype=np.float32)

    def _compute_reward(self, residual):
        imu = self.sim.get_imu()
        euler_deg = imu['euler']   # degrees
        gyro = imu['gyro']         # rad/s

        roll_rad = np.radians(euler_deg[0])
        pitch_rad = np.radians(euler_deg[1])

        ball_vel = get_ball_vel(self.sim.model, self.sim.data, self.shell_id)
        vx_err = self.controller.target_vx - ball_vel[0]
        vy_err = self.controller.target_vy - ball_vel[1]

        # --- Attitude stability (primary objective for act_dim=3) ---
        # L1 penalty: linear near 0, doesn't over-penalize large angles as much as L2
        r_att = -4.0 * (abs(roll_rad) + abs(pitch_rad))
        # Heading error term (weight 0 = legacy reward). Without it the
        # residual has no incentive on exactly the axis where the tuned PID
        # has headroom left (yaw spikes at gimbal-lock passages).
        if self.yaw_reward_weight > 0.0:
            yaw_err = wrap_angle(self.target_yaw - np.radians(euler_deg[2]))
            r_att -= self.yaw_reward_weight * abs(yaw_err)

        # --- Velocity tracking ---
        if self.act_dim == 3:
            # PID handles drive alone; RL shouldn't be rewarded/punished for vel tracking
            r_vel = 0.0
        else:
            r_vel = -1.0 * (vx_err ** 2 + vy_err ** 2)

        # --- Angular velocity damping ---
        # Normalize gyro to avoid dominating reward when angular velocities are high
        # gyro is rad/s; scale so typical values (~1 rad/s) give moderate penalty
        gyro_norm = gyro / GYRO_SCALE  # same scale as observation
        r_gyro = -0.5 * np.sum(gyro_norm ** 2)

        # --- Penalize large residual corrections ---
        # NOTE scale: with alpha_max=0.15 a useful residual |delta|~0.1 costs
        # act_penalty*0.01/channel/step, while 1 deg of yaw improvement earns
        # yaw_reward_weight*0.017/step — with the legacy coefficient (10) any
        # helpful action is net-negative, so SAC converges to zero residual.
        r_act = -self.act_penalty * np.sum(residual ** 2)

        # --- Residual smoothness: penalize sudden changes in RL output only ---
        # L2: lenient on small changes, harsh on large jumps
        d_residual = residual - self.prev_residual
        r_smooth = -self.smooth_penalty * np.sum(d_residual ** 2)

        # --- Alive bonus ---
        r_alive = 0.5
        if abs(euler_deg[0]) > FLIP_THRESHOLD or abs(euler_deg[1]) > FLIP_THRESHOLD:
            r_alive = -10.0

        return r_att + r_vel + r_gyro + r_act + r_smooth + r_alive

    def _check_done(self):
        if self.episode_step >= self.max_episode_steps:
            return True
        imu = self.sim.get_imu()
        if abs(imu['euler'][0]) > FLIP_THRESHOLD or abs(imu['euler'][1]) > FLIP_THRESHOLD:
            return True
        return False


if __name__ == '__main__':
    for adim in [3, 5]:
      for no_j in [False, True]:
        label = f"act_dim={adim}, no_joint_obs={no_j}"
        print(f"\n--- {label} ---")
        env = SphericalRobotEnv(alpha_max=0.15, warmup_steps=100, act_dim=adim,
                                terrain_prob=0.5, no_joint_obs=no_j)
        obs = env.reset(target_vx=0.5, target_vy=0.0)
        print(f"obs shape: {obs.shape} (obs_dim={env.obs_dim}), "
              f"obs range: [{obs.min():.3f}, {obs.max():.3f}]  "
              f"terrain={env._current_terrain_label}")

        total_reward = 0.0
        for i in range(200):
            action = np.random.uniform(-1, 1, size=adim)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if i % 50 == 0:
                print(f"  step={i:3d}  reward={reward:+.3f}  alpha={info['alpha']:.4f}  "
                      f"res_shape={info['residual'].shape}")
            if done:
                print(f"  Episode ended at step {i}, total_reward={total_reward:.2f}")
                break
        else:
            print(f"  Completed 200 steps, total_reward={total_reward:.2f}")
