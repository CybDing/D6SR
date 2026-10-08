"""
Robot Simulation: Main simulation interface for MuJoCo.

Provides a clean API for running spherical robot simulations with:
- 6-thruster motor control (matching STM32 motor layout)
- IMU sensor (accelerometer, gyroscope, orientation)
- Sphere-proxy ground contact with explicit rolling friction
- Stepping, state readout, and interactive viewer

Usage:
    sim = RobotSimulation()          # motors + IMU enabled by default
    sim.reset()
    sim.set_axis('roll', 0.3)        # differential roll
    for _ in range(1000):
        sim.step()
    imu = sim.get_imu()              # read IMU
    sim.run_viewer()                 # interactive viewer

Motor Layout (body frame, matches STM32 motor_calibration.c):
    M0,M1 (Blue):  +/-X pos, +Y thrust -> Z torque (Yaw)
    M2,M3 (Red):   +/-Y pos, +Z thrust -> X torque (Roll)
    M4,M5 (Green): +/-Z pos, +X thrust -> Y torque (Pitch)
"""

import platform
import time
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import mujoco


# Axis name -> (motor_a_index, motor_b_index)
_AXIS_PAIR = {
    'yaw':   (0, 1),
    'roll':  (2, 3),
    'pitch': (4, 5),
}

MJCF_PATH = Path(__file__).parent / "models" / "robot_v1_mujoco.xml"


class RobotSimulation:
    """
    MuJoCo simulation wrapper for the spherical robot.

    Builds a motor + IMU augmented model from the base MJCF at construction
    time (in-memory only, no files written).  All 6 thrusters, sphere-proxy
    contact, explicit rolling friction, and IMU sensors are injected
    automatically unless disabled.
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        motors: bool = True,
        force_max: float = 2.0,
        motor_radius: float = 0.13,
        contact_mode: str = 'sphere',
        mu_roll: float = 0.03,
        imu: bool = True,
        terrain=None,
        shell_body_name: Optional[str] = None,
        base_body_name: str = 'base_link',
        cameras: Optional[Dict[str, dict]] = None,
        xml_injectors: Optional[List] = None,
        shell_inertia: Optional[float] = None,
        inertia_overrides: Optional[Dict[str, object]] = None,
        inertia_scales: Optional[Dict[str, float]] = None,
        mass_overrides: Optional[Dict[str, float]] = None,
        pos_overrides: Optional[Dict[str, list]] = None,
        dof_damping_scale: Optional[float] = None,
        joint_frictionloss: Optional[float] = None,
        joint_damping: Optional[float] = None,
    ):
        """
        Args:
            mjcf_path:     Path to base MJCF.  None -> default model.
            motors:        Inject 6 thrusters on base_link.
            force_max:     Max thrust per motor (N).
            motor_radius:  Motor offset from center (m).
            contact_mode:  'sphere' (smooth proxy) or 'mesh' (original STL).
            mu_roll:       Explicit rolling friction coefficient (m).
            imu:           Inject accelerometer + gyro + framequat on base_link.
            terrain:       Terrain object (from terrain.py). None -> flat ground.
            shell_body_name: Name of outer shell body (default: Link3, fallback to base_body_name).
            base_body_name:  Name of base body for motors/IMU (default: base_link).
            cameras:       Dict of camera configs for video recording (see video/cameras.py).
            xml_injectors: List of callables f(root) to inject extra XML elements.
        """
        self.mjcf_path = Path(mjcf_path) if mjcf_path else MJCF_PATH
        self._motors_enabled = motors
        self._imu_enabled = imu
        self._force_max = force_max
        self._motor_radius = motor_radius
        self._contact_mode = contact_mode
        self.mu_roll = mu_roll
        self._terrain = terrain
        self._base_body_name = base_body_name
        self._shell_body_name = shell_body_name or 'Link3'
        self._cameras = cameras
        self._xml_injectors = xml_injectors or []
        self._shell_inertia = shell_inertia
        self._inertia_overrides = dict(inertia_overrides or {})
        self._inertia_scales = dict(inertia_scales or {})
        self._mass_overrides = dict(mass_overrides or {})
        self._pos_overrides = dict(pos_overrides or {})
        self._dof_damping_scale = dof_damping_scale
        self._joint_frictionloss = joint_frictionloss
        self._joint_damping = joint_damping

        # Build augmented model
        xml_str = self._build_model_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        if self._terrain is not None:
            self._terrain.populate(self.model)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._cache_model_info()
        self._shell_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self._shell_body_name)
        if self._shell_body_id < 0 and self._shell_body_name != self._base_body_name:
            self._shell_body_name = self._base_body_name
            self._shell_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self._shell_body_name)
        self._ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')

        # Optional inertia overrides (verification experiments, audit §2.1).
        # Default None → committed inertia unchanged. Used to test the effect of
        # the too-small committed inertias (shell ~10× low; inner rings dominate
        # the attitude dynamics that residual RL compensates) on control/RL.
        #   shell_inertia:      isotropic diag value for the shell body (convenience)
        #   inertia_overrides:  {body_name: scalar or [ix,iy,iz]}  (absolute)
        #   inertia_scales:     {body_name: factor}                (multiply current)
        ov = dict(self._inertia_overrides)
        if self._shell_inertia is not None:
            ov.setdefault(self._shell_body_name, float(self._shell_inertia))
        for bname, val in ov.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            old = self.model.body_inertia[bid].copy()
            self.model.body_inertia[bid] = (
                np.full(3, float(val)) if np.isscalar(val) else np.asarray(val, float))
            print(f"  [override] {bname} inertia {np.round(old,6)} -> "
                  f"{np.round(self.model.body_inertia[bid],6)}")
        for bname, factor in self._inertia_scales.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            old = self.model.body_inertia[bid].copy()
            self.model.body_inertia[bid] = old * float(factor)
            print(f"  [scale] {bname} inertia x{factor}: {np.round(old,6)} -> "
                  f"{np.round(self.model.body_inertia[bid],6)}")
        # Mass overrides (real-robot mass distribution). Inertia of a body whose
        # mass is changed but not explicitly overridden is scaled by the mass
        # ratio (same geometry, more/less material).
        for bname, new_m in self._mass_overrides.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            old_m = float(self.model.body_mass[bid])
            self.model.body_mass[bid] = float(new_m)
            if bname not in ov and bname not in self._inertia_scales and old_m > 0:
                self.model.body_inertia[bid] *= float(new_m) / old_m
            print(f"  [mass] {bname}: {old_m:.4f} -> {new_m} kg")
        # Body position overrides (e.g. move battery/ballast to set COM deviation)
        for bname, xyz in self._pos_overrides.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            old = self.model.body_pos[bid].copy()
            self.model.body_pos[bid] = np.asarray(xyz, float)
            print(f"  [pos] {bname}: {np.round(old,4)} -> {xyz}")
        # Joint (gimbal) friction realism: scale hinge damping + frictionloss.
        # Committed bearings are idealised (near-frictionless); real bearings
        # couple shell disturbances into the inner body.
        if self._dof_damping_scale is not None and self._dof_damping_scale != 1.0:
            s = float(self._dof_damping_scale)
            self.model.dof_damping[6:] *= s          # dofs 0-5 = base freejoint
            self.model.dof_frictionloss[6:] *= s
            print(f"  [damping] gimbal dof damping/frictionloss x{s}")
        # Absolute calibrated bearing values (take precedence over the scale):
        # metal deep-groove bearings incl. seal drag / slight misalignment.
        if self._joint_frictionloss is not None:
            self.model.dof_frictionloss[6:] = float(self._joint_frictionloss)
            print(f"  [bearing] frictionloss = {self._joint_frictionloss} N*m/joint")
        if self._joint_damping is not None:
            self.model.dof_damping[6:] = float(self._joint_damping)
            print(f"  [bearing] damping = {self._joint_damping} N*m*s/rad")

        # Rolling-inertia estimate for the explicit-friction impulse clamp
        # (order of magnitude suffices; see _apply_rolling_friction)
        self._roll_inertia_est = 0.7 * float(
            np.sum(self.model.body_mass)) * 0.229 ** 2

        # Cache IMU sensor addresses
        if self._imu_enabled:
            self._cache_imu_addrs()

        # Print summary
        total_mass = sum(self.model.body_mass)
        print(f"RobotSimulation ready  ({self.mjcf_path.name})")
        print(f"  bodies={self.model.nbody}  joints={self.model.njnt}  "
              f"dof={self.model.nv}  actuators={self.model.nu}")
        print(f"  mass={total_mass:.3f}kg  motors={'on' if motors else 'off'}  "
              f"imu={'on' if imu else 'off'}  contact={contact_mode}  "
              f"mu_roll={mu_roll}  shell={self._shell_body_name}")

    # ------------------------------------------------------------------
    # Model building (private)
    # ------------------------------------------------------------------

    def _build_model_xml(self) -> str:
        """Orchestrate all XML injections.  Returns XML string."""
        tree = ET.parse(str(self.mjcf_path))
        root = tree.getroot()
        self._resolve_meshdir(root)
        base = root.find(f'.//body[@name="{self._base_body_name}"]')
        if base is None:
            raise RuntimeError(f"{self._base_body_name} not found in MJCF")

        self._resolve_shell_body(root)

        if self._terrain is not None:
            self._terrain.inject_xml(root)
        if self._motors_enabled or self._contact_mode == 'sphere':
            self._inject_sphere_proxy(root)
        self._inject_visuals(root, base)
        if self._motors_enabled:
            self._inject_motors(root, base)
        if self._imu_enabled:
            self._inject_imu(root, base)

        # Video recording cameras
        if self._cameras:
            from video.cameras import inject_cameras
            inject_cameras(root, self._cameras)

        # Extra XML injectors (e.g. video visuals, trajectory markers)
        for injector in self._xml_injectors:
            injector(root)

        return ET.tostring(root, encoding='unicode', xml_declaration=True)

    def _resolve_meshdir(self, root: ET.Element):
        """PORTABILITY — make mesh loading independent of where the repo lives.

        The committed MJCF (generated by robot_loader.py) stores ``meshdir`` as a
        path RELATIVE to its own directory so the repo survives being moved or
        cloned. Two issues are handled here:

          1. ``from_xml_string()`` (used in ``__init__``) resolves a relative
             meshdir against the CWD, not the XML file — so we convert it to an
             absolute path anchored at the XML's location.
          2. SELF-HEAL: an older XML may still carry a STALE ABSOLUTE meshdir
             baked in at generation time on a different machine/path. If the
             recorded directory does not exist, fall back to the ``meshes``
             folder that ships next to this source tree.

        Net effect: `python ...` just works after a fresh `git clone`, with no
        manual `robot_loader.py` regeneration step.
        """
        compiler = root.find('compiler')
        if compiler is None:
            return
        repo_meshes = (Path(__file__).resolve().parent.parent
                       / 'robot_v1' / 'meshes')
        md = compiler.get('meshdir')
        if md and not Path(md).is_absolute():
            abs_md = (self.mjcf_path.resolve().parent / md).resolve()
        elif md:
            abs_md = Path(md)
        else:
            abs_md = repo_meshes
        if not abs_md.exists() and repo_meshes.exists():
            abs_md = repo_meshes
        compiler.set('meshdir', str(abs_md))

    def _resolve_shell_body(self, root: ET.Element):
        """Resolve shell body name, falling back to base body if needed."""
        if root.find(f'.//body[@name="{self._shell_body_name}"]') is None:
            if root.find(f'.//body[@name="{self._base_body_name}"]') is None:
                raise RuntimeError("No valid shell/base body found in MJCF")
            self._shell_body_name = self._base_body_name

    def _inject_sphere_proxy(self, root: ET.Element):
        """Replace shell mesh collision with a smooth sphere proxy (condim=4)."""
        shell_name = self._shell_body_name
        if self._contact_mode != 'sphere':
            # Still downgrade mesh to condim=4
            for body in root.findall(f'.//body[@name="{shell_name}"]'):
                for geom in body.findall('geom'):
                    geom.set('condim', '4')
            # Also downgrade ground
            ground = root.find('.//geom[@name="ground"]')
            if ground is not None:
                ground.set('condim', '4')
            return

        # Measure shell radius by settling a temporary model.
        # PORTABILITY: load from a meshdir-corrected copy of the raw MJCF rather
        # than the on-disk path, so a stale/absolute meshdir in the committed
        # file cannot break this measurement on a freshly-cloned checkout.
        _raw = ET.parse(str(self.mjcf_path)).getroot()
        self._resolve_meshdir(_raw)
        _tmp = mujoco.MjModel.from_xml_string(ET.tostring(_raw, encoding='unicode'))
        _tmp_d = mujoco.MjData(_tmp)
        _shell_id = mujoco.mj_name2id(_tmp, mujoco.mjtObj.mjOBJ_BODY, shell_name)
        if _shell_id < 0:
            return
        _com_offset = _tmp.body_ipos[_shell_id].copy()
        mujoco.mj_forward(_tmp, _tmp_d)
        for _ in range(5000):
            mujoco.mj_step(_tmp, _tmp_d)
        _ground_z = -0.5
        shell_r = float(_tmp_d.xipos[_shell_id][2] - _ground_z)
        if shell_r < 0.05:
            shell_r = 0.2293  # fallback

        # Add a low-specular material so shell mesh doesn't blow out to solid
        # blue under directional lighting.
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        shell_mat = ET.SubElement(asset, 'material')
        shell_mat.set('name', 'shell_mat')
        shell_mat.set('specular', '0.04')
        shell_mat.set('shininess', '0.01')
        shell_mat.set('reflectance', '0.0')

        for body in root.findall(f'.//body[@name="{shell_name}"]'):
            for geom in body.findall('geom'):
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                geom.set('rgba', '0.35 0.50 0.85 0.20')
                geom.set('material', 'shell_mat')
            coll = ET.SubElement(body, 'geom')
            coll.set('name', 'shell_contact')
            coll.set('type', 'sphere')
            coll.set('size', f'{shell_r:.4f}')
            coll.set('pos', f'{_com_offset[0]:.6f} {_com_offset[1]:.6f} {_com_offset[2]:.6f}')
            coll.set('rgba', '0 0 0 0')
            coll.set('contype', '1')
            coll.set('conaffinity', '1')
            coll.set('friction', '1.0 0.02 0.001')
            coll.set('condim', '4')
            coll.set('solref', '0.02 1')
            coll.set('solimp', '0.9 0.95 0.01')

        # Ground must also be condim=4
        ground = root.find('.//geom[@name="ground"]')
        if ground is not None:
            ground.set('condim', '4')

    def _inject_visuals(self, root: ET.Element, base: ET.Element):
        """Add lighting and RGB axis capsules on base_link."""
        # Headlight + sky
        visual_elem = root.find('visual')
        if visual_elem is None:
            visual_elem = ET.SubElement(root, 'visual')
        hl = ET.SubElement(visual_elem, 'headlight')
        hl.set('ambient', '0.55 0.55 0.55')
        hl.set('diffuse', '0.85 0.85 0.85')
        hl.set('specular', '0.15 0.15 0.15')
        hl.set('active', '1')
        sky = ET.SubElement(visual_elem, 'rgba')
        sky.set('haze', '0.90 0.89 0.86 1')  # matches ground colour

        # Fog to hide ground / skybox seam.
        # fogstart/fogend are in *multiples of model extent*, NOT meters.
        # With plane size=50 → extent ≈ 50 m, so 0.4 × 50 = 20 m, 1.5 × 50 = 75 m.
        mapelem = ET.SubElement(visual_elem, 'map')
        mapelem.set('fogstart', '0.4')
        mapelem.set('fogend', '1.5')

        # Overhead light
        worldbody = root.find('.//worldbody')
        if worldbody is not None:
            sun = ET.Element('light')
            sun.set('name', 'sun')
            sun.set('pos', '0 0 2')
            sun.set('dir', '0 -0.3 -1')
            sun.set('diffuse', '0.28 0.28 0.25')
            sun.set('specular', '0.04 0.04 0.04')
            sun.set('castshadow', 'true')
            worldbody.insert(0, sun)

        # RGB axis capsules
        axes = [
            ("axis_x", "0 0 0 0.14 0 0",  "1.0 0.15 0.15 1"),
            ("axis_y", "0 0 0 0 0.14 0",  "0.1 1.0  0.15 1"),
            ("axis_z", "0 0 0 0 0 0.14",  "0.15 0.45 1.0 1"),
        ]
        for name, fromto, rgba in axes:
            ax = ET.SubElement(base, 'geom')
            ax.set('name', name)
            ax.set('type', 'capsule')
            ax.set('fromto', fromto)
            ax.set('size', '0.005')
            ax.set('rgba', rgba)
            ax.set('contype', '0')
            ax.set('conaffinity', '0')

    def _inject_motors(self, root: ET.Element, base: ET.Element):
        """Inject 6 thruster sites + actuators on base_link."""
        d = self._motor_radius
        F = self._force_max

        motor_defs = [
            ("motor0_yaw_n",   f"{+d} 0 0", f"0 {F} 0 0 0 0", "0.20 0.20 0.90 0.9"),
            ("motor1_yaw_p",   f"{-d} 0 0", f"0 {F} 0 0 0 0", "0.35 0.35 1.00 0.9"),
            ("motor2_roll_n",  f"0 {+d} 0", f"0 0 {F} 0 0 0", "0.90 0.20 0.20 0.9"),
            ("motor3_roll_p",  f"0 {-d} 0", f"0 0 {F} 0 0 0", "1.00 0.40 0.40 0.9"),
            ("motor4_pitch_n", f"0 0 {+d}", f"{F} 0 0 0 0 0", "0.20 0.85 0.20 0.9"),
            ("motor5_pitch_p", f"0 0 {-d}", f"{F} 0 0 0 0 0", "0.35 1.00 0.35 0.9"),
        ]

        # Sites
        for name, pos, _, rgba in motor_defs:
            site = ET.SubElement(base, 'site')
            site.set('name', name)
            site.set('pos', pos)
            site.set('size', '0.006')
            site.set('rgba', rgba)

        # Actuators
        actuator_elem = root.find('actuator')
        if actuator_elem is None:
            actuator_elem = ET.SubElement(root, 'actuator')
        for name, _, gear, _ in motor_defs:
            act = ET.SubElement(actuator_elem, 'general')
            act.set('name', name)
            act.set('site', name)
            act.set('gear', gear)
            act.set('ctrlrange', '-1 1')
            act.set('ctrllimited', 'true')

    def _inject_imu(self, root: ET.Element, base: ET.Element):
        """Inject IMU sensors (accel, gyro, framequat) on base_link."""
        # Site at body origin, no rotation -> aligned with body frame
        site = ET.SubElement(base, 'site')
        site.set('name', 'imu_site')
        site.set('pos', '0 0 0')
        site.set('size', '0.003')
        site.set('rgba', '1 1 0 0.5')

        # Sensor section
        sensor_elem = root.find('sensor')
        if sensor_elem is None:
            sensor_elem = ET.SubElement(root, 'sensor')

        accel = ET.SubElement(sensor_elem, 'accelerometer')
        accel.set('name', 'imu_accel')
        accel.set('site', 'imu_site')

        gyro = ET.SubElement(sensor_elem, 'gyro')
        gyro.set('name', 'imu_gyro')
        gyro.set('site', 'imu_site')

        quat = ET.SubElement(sensor_elem, 'framequat')
        quat.set('name', 'imu_quat')
        quat.set('objtype', 'site')
        quat.set('objname', 'imu_site')

    def _cache_imu_addrs(self):
        """Cache sensor data addresses for fast get_imu() reads."""
        def _addr(name):
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise RuntimeError(f"Sensor '{name}' not found")
            return self.model.sensor_adr[sid], self.model.sensor_dim[sid]

        self._accel_adr, self._accel_dim = _addr('imu_accel')
        self._gyro_adr, self._gyro_dim = _addr('imu_gyro')
        self._quat_adr, self._quat_dim = _addr('imu_quat')

    # ------------------------------------------------------------------
    # Model info cache
    # ------------------------------------------------------------------

    def _cache_model_info(self):
        """Cache joint names and IDs."""
        self.joint_names = []
        self.joint_ids = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names.append(name)
                self.joint_ids[name] = i

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time(self) -> float:
        """Current simulation time (s)."""
        return self.data.time

    @property
    def dt(self) -> float:
        """Simulation timestep (s)."""
        return self.model.opt.timestep

    # ------------------------------------------------------------------
    # Motor control
    # ------------------------------------------------------------------

    def set_ctrl(self, ctrl: Union[List[float], np.ndarray]):
        """Set all 6 motor controls at once.  Values in [-1, 1]."""
        self.data.ctrl[:len(ctrl)] = ctrl

    def set_motor(self, idx: int, val: float):
        """Set a single motor by index (0-5).  Value in [-1, 1]."""
        self.data.ctrl[idx] = val

    def set_axis(self, axis: str, val: float, mode: str = 'differential'):
        """Set a motor pair by axis name.

        Args:
            axis: 'yaw', 'roll', or 'pitch'
            val:  Control magnitude in [-1, 1]
            mode: 'differential' -> pure torque (mA=+val, mB=-val)
                  'common'       -> net force   (mA=+val, mB=+val)
        """
        a, b = _AXIS_PAIR[axis]
        if mode == 'differential':
            self.data.ctrl[a] = val
            self.data.ctrl[b] = -val
        elif mode == 'common':
            self.data.ctrl[a] = val
            self.data.ctrl[b] = val
        else:
            raise ValueError(f"Unknown mode: {mode!r}  (use 'differential' or 'common')")

    def get_ctrl(self) -> np.ndarray:
        """Return a copy of the current control vector."""
        return self.data.ctrl.copy()

    # ------------------------------------------------------------------
    # IMU readout
    # ------------------------------------------------------------------

    def get_imu(self) -> Dict[str, np.ndarray]:
        """Read IMU sensor data.

        Returns dict with:
            'accel': [ax, ay, az] m/s^2 in body frame (includes gravity)
            'gyro':  [gx, gy, gz] rad/s in body frame
            'quat':  [w, x, y, z] orientation quaternion (world frame)
            'euler': [roll, pitch, yaw] degrees (derived from quat)
        """
        if not self._imu_enabled:
            raise RuntimeError("IMU not enabled (pass imu=True to constructor)")

        accel = self.data.sensordata[
            self._accel_adr:self._accel_adr + self._accel_dim].copy()
        gyro = self.data.sensordata[
            self._gyro_adr:self._gyro_adr + self._gyro_dim].copy()
        quat = self.data.sensordata[
            self._quat_adr:self._quat_adr + self._quat_dim].copy()

        # Quaternion -> Euler (roll, pitch, yaw)
        euler = self._quat_to_euler(quat)

        return {
            'accel': accel,
            'gyro': gyro,
            'quat': quat,
            'euler': euler,
        }

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
        w, x, y, z = q
        # Roll (X)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr, cosr)
        # Pitch (Y)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (Z)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny, cosy)
        return np.array([roll, pitch, yaw]) * 180 / np.pi

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, ctrl: Optional[Union[List[float], np.ndarray]] = None):
        """Advance simulation by one timestep.

        Applies explicit rolling friction before mj_step.

        Args:
            ctrl: If provided, set motor controls before stepping.
        """
        if ctrl is not None and self.model.nu > 0:
            self.data.ctrl[:len(ctrl)] = ctrl

        self._apply_rolling_friction()
        mujoco.mj_step(self.model, self.data)

    def _apply_rolling_friction(self):
        """Apply explicit rolling resistance torque to shell body (Stribeck model).

        Replaces condim=6 built-in rolling friction to avoid friction-cone
        chattering.  See MEMORY.md section 3a for full derivation.
        """
        sid = self._shell_body_id
        gid = self._ground_geom_id
        if sid < 0 or gid < 0:
            return

        # Clear previous explicit torque
        self.data.xfrc_applied[sid, 3:] = 0

        if self.mu_roll <= 0:
            return

        # Sum normal forces from ground contacts
        fn_total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                fn_total += f[0]

        if fn_total < 0.01:
            return

        # Angular velocity of shell body in world frame
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, sid, vel, 0)
        omega = vel[:3]

        # Rolling = rotation around ground-parallel axes (X, Y)
        omega_roll = np.array([omega[0], omega[1], 0.0])
        omega_mag = np.linalg.norm(omega_roll)

        # Stribeck smoothing: viscous below threshold, Coulomb above
        v_stribeck = 0.1  # rad/s
        tau = -self.mu_roll * fn_total * omega_roll / max(omega_mag, v_stribeck)

        # Impulse clamp: an explicit friction torque must not reverse the
        # rolling rate within one step (sign-flipping stick-slip pumps energy
        # and diverges at large mu_roll): |tau| <= 0.5*I*|omega|/dt.
        tau_mag = np.linalg.norm(tau)
        tau_max = (0.5 * self._roll_inertia_est * omega_mag
                   / self.model.opt.timestep)
        if tau_mag > tau_max > 0:
            tau *= tau_max / tau_mag

        self.data.xfrc_applied[sid, 3:] = tau

    # ------------------------------------------------------------------
    # State readout
    # ------------------------------------------------------------------

    def reset(self, qpos: Optional[np.ndarray] = None,
              qvel: Optional[np.ndarray] = None):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:len(qpos)] = qpos
        if qvel is not None:
            self.data.qvel[:len(qvel)] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            'time': self.data.time,
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'xpos': self.data.xpos.copy(),
            'xquat': self.data.xquat.copy(),
        }

    def get_joint_state(self, joint_name: str) -> Tuple[float, float]:
        """Get (position, velocity) of a named joint."""
        joint_id = self.joint_ids.get(joint_name)
        if joint_id is None:
            raise ValueError(f"Unknown joint: {joint_name}")
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        return self.data.qpos[qpos_adr], self.data.qvel[qvel_adr]

    def get_contact_info(self) -> Dict[str, Any]:
        """Get ground contact information for shell body.

        Returns dict with:
            'in_contact': bool
            'normal':     float  (N, positive = support)
            'slide':      float  (N, tangential friction)
            'torsion':    float  (N*m, torsional moment)
            'roll_torque': [tx, ty, tz]  (N*m, explicit rolling friction)
            'gap_mm':     float  (contact gap in mm)
        """
        gid = self._ground_geom_id
        sid = self._shell_body_id
        result = {
            'in_contact': False,
            'normal': 0.0,
            'slide': 0.0,
            'torsion': 0.0,
            'roll_torque': np.zeros(3),
            'gap_mm': 0.0,
        }

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                result['in_contact'] = True
                result['normal'] = f[0]
                result['slide'] = np.hypot(f[1], f[2])
                result['torsion'] = abs(f[3])
                result['gap_mm'] = c.dist * 1000
                break

        if sid >= 0:
            result['roll_torque'] = self.data.xfrc_applied[sid, 3:].copy()

        return result

    # ------------------------------------------------------------------
    # Force / damping helpers (preserved)
    # ------------------------------------------------------------------

    def apply_force(self, body_name: str, force: np.ndarray,
                    torque: Optional[np.ndarray] = None):
        """Apply external force/torque to a body (world frame)."""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Unknown body: {body_name}")
        if torque is None:
            torque = np.zeros(3)
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = torque

    def clear_forces(self):
        """Clear all external forces."""
        self.data.xfrc_applied[:] = 0

    # def set_joint_damping(self, joint_name: str, damping: float):
    #     """Dynamically adjust joint damping."""
    #     joint_id = self.joint_ids.get(joint_name)
    #     if joint_id is None:
    #         raise ValueError(f"Unknown joint: {joint_name}")
    #     dof_adr = self.model.jnt_dofadr[joint_id]
    #     self.model.dof_damping[dof_adr] = damping

    # def apply_joint_kick(self, magnitude: float = 0.6,
    #                      pattern: Optional[Tuple[float, float, float]] = None):
    #     """Apply initial joint velocities to trigger motion."""
    #     if magnitude == 0:
    #         return
    #     if pattern is None:
    #         pattern = (1.0, -0.7, 0.4)
    #     for joint_name, scale in zip(["joint1", "joint2", "joint3"], pattern):
    #         joint_id = self.joint_ids.get(joint_name)
    #         if joint_id is None:
    #             continue
    #         dof_adr = self.model.jnt_dofadr[joint_id]
    #         self.data.qvel[dof_adr] = magnitude * scale

    # ------------------------------------------------------------------
    # Simulation loops
    # ------------------------------------------------------------------

    def simulate(self, duration: float, dt: Optional[float] = None) -> list:
        """Run simulation for a duration and collect trajectory.

        Returns list of state dicts (one per step).
        """
        if dt is not None:
            self.model.opt.timestep = dt
        steps = int(duration / self.model.opt.timestep)
        trajectory = []
        for _ in range(steps):
            self.step()
            trajectory.append(self.get_state())
        return trajectory

    def run_viewer(self):
        """Launch interactive MuJoCo viewer with real-time stepping.

        Rolling friction is applied automatically each step.
        On macOS, must be run with `mjpython`.
        """
        if platform.system() == 'Darwin':
            print("\nNote: On macOS, run with:  mjpython simulation.py")

        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                print("Viewer launched. Close window to exit.")
                while viewer.is_running():
                    step_start = time.perf_counter()
                    self.step()
                    print(self.get_imu())
                    viewer.sync()
                    elapsed = time.perf_counter() - step_start
                    sleep_time = self.model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            import traceback
            print(f"\nViewer error: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("\nFalling back to headless...")
            self._run_headless()

    def _run_headless(self, steps: int = 1000, print_interval: int = 100):
        """Run simulation without visualization."""
        print(f"\nHeadless simulation ({steps} steps)...")
        self.reset()
        for i in range(steps):
            self.step()
            if i % print_interval == 0:
                state = self.get_state()
                pos = state['xpos'][1]
                print(f"  Step {i:4d}: base_pos = "
                      f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print("Done!")


def main():
    """Demo: create sim, settle, read IMU, apply roll, read IMU, launch viewer."""
    sim = RobotSimulation(mu_roll=0.01)
    sim.reset()

    # --- Settle ---
    print("\nSettling (3s)...")
    settle_steps = int(3.0 / sim.dt)
    for _ in range(settle_steps):
        sim.step()

    # --- IMU at rest ---
    imu = sim.get_imu()
    print(f"\nIMU at rest (t={sim.time:.2f}s):")
    print(f"  accel = [{imu['accel'][0]:+7.3f}, {imu['accel'][1]:+7.3f}, {imu['accel'][2]:+7.3f}] m/s^2")
    print(f"  gyro  = [{imu['gyro'][0]:+7.4f}, {imu['gyro'][1]:+7.4f}, {imu['gyro'][2]:+7.4f}] rad/s")
    print(f"  euler = [{imu['euler'][0]:+6.2f}, {imu['euler'][1]:+6.2f}, {imu['euler'][2]:+6.2f}] deg")

    # --- Apply roll ---
    print("\nApplying roll (ctrl=0.5) for 2s...")
    sim.set_axis('pitch', 0.00, mode='common')
    roll_steps = int(10.0 / sim.dt)
    for _ in range(roll_steps):
        sim.step()

    imu = sim.get_imu()
    contact = sim.get_contact_info()
    print(f"\nIMU after roll (t={sim.time:.2f}s):")
    print(f"  accel = [{imu['accel'][0]:+7.3f}, {imu['accel'][1]:+7.3f}, {imu['accel'][2]:+7.3f}] m/s^2")
    print(f"  gyro  = [{imu['gyro'][0]:+7.4f}, {imu['gyro'][1]:+7.4f}, {imu['gyro'][2]:+7.4f}] rad/s")
    print(f"  euler = [{imu['euler'][0]:+6.2f}, {imu['euler'][1]:+6.2f}, {imu['euler'][2]:+6.2f}] deg")
    print(f"  contact: normal={contact['normal']:.2f}N  slide={contact['slide']:.3f}N  "
          f"in_contact={contact['in_contact']}")

    # --- Stop motors, launch viewer ---
    sim.set_ctrl([0, 0, 0, 0, 0, 0])
    print("\nCtrl=[0,0,0,0,0,0] -- launching viewer...")
    sim.run_viewer()


if __name__ == "__main__":
    main()
