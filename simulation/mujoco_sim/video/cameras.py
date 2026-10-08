"""
Camera definitions and XML injection for video recording.

Cameras are injected into the MJCF XML before model compilation.
The follow camera tracks the shell_contact geom (true ball center)
rather than the Link3 body origin (which is offset ~20cm).
"""

import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
import mujoco


# ── Camera presets ───────────────────────────────────────────────────────

def default_cameras(target_body='base_link'):
    """Return dict of camera configs for all standard angles."""
    return {
        'overview': {
            'pos': '0 -3 2.5',
            'xyaxes': '1 0 0 0 0.5 1',
            'fovy': '50',
        },
        'follow': {
            # Initial pos; updated at runtime by FollowCamera
            'pos': '-1.5 -1.0 0.6',
            'mode': 'targetbody',
            'target': target_body,
            'fovy': '40',
        },
        'side': {
            'pos': '3 -2.5 0.5',
            'xyaxes': '1 0 0 0 0.2 1',
            'fovy': '45',
        },
        'topdown': {
            # Updated at runtime by FollowCamera for tracking
            'pos': '0 0 5',
            'xyaxes': '1 0 0 0 1 0',
            'fovy': '50',
        },
    }


def trajectory_cameras(radius=5.0, target_body='base_link'):
    """Cameras tuned for circular trajectory (centred at origin)."""
    r = radius
    cams = default_cameras(target_body)
    # Overview: see full circle with some margin
    h = r * 0.8 + 2
    d = r * 0.6 + 3
    cams['overview']['pos'] = f'0 {-d:.1f} {h:.1f}'
    cams['overview']['fovy'] = '55'
    # Topdown: just enough to frame the circle
    cams['topdown']['pos'] = f'0 0 {r * 1.6 + 2:.1f}'
    cams['topdown']['fovy'] = '55'
    return cams


def terrain_cameras(zone_boundary_x=8.0, target_body='base_link'):
    """Cameras tuned for terrain switching (rolling in +X direction)."""
    cams = default_cameras(target_body)
    # Overview: positioned to see the first ~25m of travel, closer
    cams['overview']['pos'] = '12 -6 4'
    cams['overview']['fovy'] = '55'
    # Side: at first zone boundary, close side view
    cams['side']['pos'] = f'{zone_boundary_x} -3 0.5'
    cams['side']['xyaxes'] = '1 0 0 0 0.15 1'
    cams['side']['fovy'] = '45'
    # Topdown: centred on travel path
    cams['topdown']['pos'] = '15 0 10'
    cams['topdown']['fovy'] = '55'
    return cams


def disturbance_cameras(target_body='base_link'):
    """Cameras tuned for stationary disturbance rejection (close up)."""
    cams = default_cameras(target_body)
    cams['overview']['pos'] = '0 -2.5 1.8'
    cams['overview']['fovy'] = '45'
    cams['side']['pos'] = '2 -1.8 0.3'
    cams['side']['fovy'] = '40'
    cams['topdown']['pos'] = '0 0 4'
    cams['topdown']['fovy'] = '45'
    return cams


# ── XML injection ────────────────────────────────────────────────────────

def inject_cameras(root: ET.Element, camera_configs: dict):
    """Add <camera> elements to <worldbody> in the XML tree."""
    worldbody = root.find('.//worldbody')
    if worldbody is None:
        return
    for name, cfg in camera_configs.items():
        cam = ET.SubElement(worldbody, 'camera')
        cam.set('name', name)
        cam.set('pos', cfg['pos'])
        if 'xyaxes' in cfg:
            cam.set('xyaxes', cfg['xyaxes'])
        if 'euler' in cfg:
            cam.set('euler', cfg['euler'])
        cam.set('fovy', cfg.get('fovy', '60'))
        if 'mode' in cfg:
            cam.set('mode', cfg['mode'])
        if 'target' in cfg:
            cam.set('target', cfg['target'])


# ── Runtime follow-camera updater ────────────────────────────────────────

class FollowCamera:
    """Smoothly track the shell center (shell_contact geom) each render frame.

    Uses the shell_contact geom world position (true ball center) rather than
    the Link3 body origin, which is offset ~20cm from the actual sphere center
    due to the SolidWorks assembly origin.
    """

    def __init__(self, model, data, cam_name='follow',
                 geom_name='shell_contact', offset=np.array([-1.5, -1.0, 0.6]),
                 smooth_alpha=0.05, lock_height=True,
                 fixed_z: Optional[float] = None):
        self.model = model
        self.data = data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        self.offset = offset.copy()
        self.alpha = smooth_alpha
        self.lock_height = lock_height
        self.fixed_z = fixed_z
        self._smooth_pos = None

    def update(self):
        """Call once per render frame to update camera position."""
        if self.cam_id < 0 or self.geom_id < 0:
            return
        # shell_contact geom xpos = true ball center in world frame
        ball_center = self.data.geom_xpos[self.geom_id]
        target = ball_center + self.offset
        if self.lock_height:
            if self.fixed_z is None:
                self.fixed_z = float(target[2])
            target[2] = self.fixed_z
        if self._smooth_pos is None:
            self._smooth_pos = target.copy()
        else:
            self._smooth_pos += self.alpha * (target - self._smooth_pos)
            if self.lock_height:
                self._smooth_pos[2] = self.fixed_z
        self.model.cam_pos[self.cam_id] = self._smooth_pos
