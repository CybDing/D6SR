"""
Enhanced visuals for video recording: skybox, ground textures, lighting,
trajectory markers, actual-path trail, and terrain-aware marker placement.

These are XML injection functions called during model building,
alongside the existing _inject_visuals() in simulation.py.
"""

import xml.etree.ElementTree as ET
import numpy as np


# ======================================================================
# XML injection (pre-compilation)
# ======================================================================

def inject_video_visuals(root: ET.Element, ground_checker=True,
                          offwidth=1920, offheight=1080,
                          white_bg=False, fog_extent=None):
    """Add skybox, enhanced lighting, offscreen buffer, and optional checker ground.

    Args:
        white_bg: If True, use white sky/ground/fog for clean paper figures.
        fog_extent: If set, override mjModel.stat.extent (meters) so fog
                    distances are predictable regardless of geometry size.
                    Useful for terrain scenes where auto-computed extent is
                    inflated by the ground_far plane.
    """
    visual_elem = root.find('visual')
    if visual_elem is None:
        visual_elem = ET.SubElement(root, 'visual')
    gl = visual_elem.find('global')
    if gl is None:
        gl = ET.SubElement(visual_elem, 'global')
    gl.set('offwidth', str(offwidth))
    gl.set('offheight', str(offheight))

    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')

    # ── Skybox ────────────────────────────────────────────────────────────
    sky_tex = ET.SubElement(asset, 'texture')
    sky_tex.set('name', 'sky_tex')
    sky_tex.set('type', 'skybox')
    sky_tex.set('builtin', 'gradient')
    if white_bg:
        sky_tex.set('rgb1', '1 1 1')
        sky_tex.set('rgb2', '1 1 1')
    else:
        # rgb2 (horizon) must match ground & haze for seamless transition
        sky_tex.set('rgb1', '0.55 0.65 0.88')
        sky_tex.set('rgb2', '0.90 0.89 0.86')
    sky_tex.set('width', '512')
    sky_tex.set('height', '3072')

    # ── Override haze + fog ───────────────────────────────────────────────
    haze = visual_elem.find('rgba')
    if haze is None:
        haze = ET.SubElement(visual_elem, 'rgba')

    mapelem = visual_elem.find('map')
    if mapelem is None:
        mapelem = ET.SubElement(visual_elem, 'map')

    # Explicit model extent makes fog distances predictable
    if fog_extent is not None:
        stat = root.find('statistic')
        if stat is None:
            stat = ET.SubElement(root, 'statistic')
        stat.set('extent', str(fog_extent))

    if white_bg:
        haze.set('haze', '1 1 1 1')
        # Disable fog for white scene (sky=white, ground=white, no seam)
        mapelem.set('fogstart', '100')
        mapelem.set('fogend', '200')
    else:
        # Haze = ground colour = skybox horizon for seamless fog transition
        haze.set('haze', '0.90 0.89 0.86 1')
        # Fog hides heightfield edges.  fogstart/fogend are multiples of
        # stat.extent.  With fog_extent=100: start=20m, end=70m.
        mapelem.set('fogstart', '0.2')
        mapelem.set('fogend', '0.7')

    # ── Headlight override ─────────────────────────────────────────────────
    hl = visual_elem.find('headlight')
    if hl is not None:
        if white_bg:
            # Pure ambient — fully uniform, no directional component.
            hl.set('ambient', '1.0 1.0 1.0')
            hl.set('diffuse', '0.0 0.0 0.0')
            hl.set('specular', '0.0 0.0 0.0')
        else:
            # High ambient so no surface is dark, with a mild directional
            # component to preserve some depth perception on terrain.
            hl.set('ambient', '0.90 0.90 0.90')
            hl.set('diffuse', '0.25 0.25 0.25')
            hl.set('specular', '0.03 0.03 0.03')

    # Remove world lights (sun etc.) — they cause harsh dark patches on
    # terrain faces that point away from the light direction.
    worldbody = root.find('.//worldbody')
    if worldbody is not None:
        if white_bg:
            for light in list(worldbody.findall('light')):
                worldbody.remove(light)
        else:
            # Dim existing lights rather than removing (keep subtle shadows)
            for light in list(worldbody.findall('light')):
                light.set('diffuse', '0.08 0.08 0.08')
                light.set('specular', '0.01 0.01 0.01')

    # ── Checker / white ground ────────────────────────────────────────────
    if ground_checker or white_bg:
        ground = root.find('.//geom[@name="ground"]')
        if white_bg:
            white_tex = ET.SubElement(asset, 'texture')
            white_tex.set('name', 'white_tex')
            white_tex.set('type', '2d')
            white_tex.set('builtin', 'flat')
            white_tex.set('rgb1', '0.97 0.97 0.97')
            white_tex.set('width', '64')
            white_tex.set('height', '64')
            white_mat = ET.SubElement(asset, 'material')
            white_mat.set('name', 'white_mat')
            white_mat.set('texture', 'white_tex')
            white_mat.set('texrepeat', '1 1')
            white_mat.set('reflectance', '0.0')
            if ground is not None and ground.get('type', 'plane') == 'plane':
                ground.set('material', 'white_mat')
                ground.set('rgba', '0.97 0.97 0.97 1')
        elif ground_checker:
            checker_tex = ET.SubElement(asset, 'texture')
            checker_tex.set('name', 'grid_tex')
            checker_tex.set('type', '2d')
            checker_tex.set('builtin', 'checker')
            checker_tex.set('rgb1', '0.88 0.87 0.83')
            checker_tex.set('rgb2', '0.78 0.77 0.73')
            checker_tex.set('width', '512')
            checker_tex.set('height', '512')
            checker_mat = ET.SubElement(asset, 'material')
            checker_mat.set('name', 'grid_mat')
            checker_mat.set('texture', 'grid_tex')
            checker_mat.set('texrepeat', '20 20')
            checker_mat.set('reflectance', '0.08')
            if ground is not None and ground.get('type', 'plane') == 'plane':
                ground.set('material', 'grid_mat')

    # ── Fill light — only when headlight ambient is low (no video override) ─
    # With the high-ambient headlight above, an extra fill light is not
    # needed and would re-introduce directional dark patches.


# ======================================================================
# Dynamic markers (pre-allocate hidden sites, reveal at runtime)
# ======================================================================

def inject_marker_sites(root: ET.Element, prefix: str, n: int,
                         size: float, rgba: str):
    """Pre-allocate n hidden sphere sites with given prefix in worldbody.

    Creates an emissive material so markers are self-illuminated and
    appear the same colour regardless of lighting direction.
    """
    # Create emissive material for this marker set
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    mat_name = f'{prefix}mat'
    mat = ET.SubElement(asset, 'material')
    mat.set('name', mat_name)
    mat.set('emission', '1.0')
    mat.set('specular', '0.0')
    mat.set('shininess', '0.0')
    mat.set('reflectance', '0.0')

    worldbody = root.find('.//worldbody')
    if worldbody is None:
        return
    for i in range(n):
        site = ET.SubElement(worldbody, 'site')
        site.set('name', f'{prefix}{i}')
        site.set('pos', '0 0 -10')  # hidden below ground
        site.set('size', str(size))
        site.set('rgba', rgba)
        site.set('type', 'sphere')
        site.set('material', mat_name)


def inject_ref_and_trail_sites(root: ET.Element,
                                n_ref: int = 600, n_trail: int = 500,
                                ref_size: float = 0.022,
                                ref_rgba: str = '0.95 0.05 0.15 0.85',
                                trail_size: float = 0.014,
                                trail_rgba: str = '0.05 0.55 0.95 0.90'):
    """Pre-allocate both reference trajectory and actual trail marker sites."""
    inject_marker_sites(root, 'ref_traj_', n_ref, ref_size, ref_rgba)
    inject_marker_sites(root, 'trail_', n_trail, trail_size, trail_rgba)


class RefTrajectoryDrawer:
    """Dynamically reveals reference trajectory markers with lookahead.

    Stamps dots from the trajectory start up to current_time + lookahead,
    so the upcoming path is always visible near the robot. Markers placed
    once stay permanently.

    Args:
        height_fn: Optional callable(x, y) -> z for terrain-aware placement.
                   If None, uses flat ground_z.
    """

    def __init__(self, model, trajectory, prefix='ref_traj_', n_markers=600,
                 ground_z=-0.46, height_fn=None, spacing_m=0.18):
        import mujoco
        self.model = model
        self.traj = trajectory
        self.ground_z = ground_z
        self.height_fn = height_fn
        self.spacing_m = spacing_m  # meters between dots
        self.site_ids = []
        for i in range(n_markers):
            sid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f'{prefix}{i}')
            if sid >= 0:
                self.site_ids.append(sid)
        self.next_idx = 0
        self._next_t = 0.0
        # Compute time spacing from speed
        vx, vy = trajectory.velocity(0.0)
        speed = max(np.hypot(vx, vy), 0.1)
        self._dt = spacing_m / speed

    def update(self, sim_t, lookahead=6.0):
        """Reveal markers up to sim_t + lookahead seconds."""
        t_end = sim_t + lookahead
        while self._next_t <= t_end and self.next_idx < len(self.site_ids):
            x, y = self.traj.position(self._next_t)
            if self.height_fn is not None:
                z = self.height_fn(x, y) + 0.025
            else:
                z = self.ground_z
            sid = self.site_ids[self.next_idx]
            self.model.site_pos[sid][0] = x
            self.model.site_pos[sid][1] = y
            self.model.site_pos[sid][2] = z
            self.next_idx += 1
            self._next_t += self._dt


class TrailDrawer:
    """Stamps trail dots at the ball's ground-contact position.

    Call stamp() periodically (e.g. every 0.3s) during the main loop.
    """

    def __init__(self, model, data, geom_name='shell_contact',
                 prefix='trail_', n_trail=500):
        import mujoco
        self.model = model
        self.data = data
        self.geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        self.radius = (model.geom_size[self.geom_id][0]
                        if self.geom_id >= 0 else 0.23)
        self.site_ids = []
        for i in range(n_trail):
            sid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f'{prefix}{i}')
            if sid >= 0:
                self.site_ids.append(sid)
        self.next_idx = 0

    def stamp(self):
        """Place the next trail marker at current ball ground-contact point."""
        if self.geom_id < 0 or self.next_idx >= len(self.site_ids):
            return
        ball_pos = self.data.geom_xpos[self.geom_id]
        sid = self.site_ids[self.next_idx]
        self.model.site_pos[sid][0] = ball_pos[0]
        self.model.site_pos[sid][1] = ball_pos[1]
        self.model.site_pos[sid][2] = ball_pos[2] - self.radius + 0.012
        self.next_idx += 1


# ======================================================================
# Post-compilation: adjust marker z to terrain surface
# ======================================================================

def make_terrain_height_fn(model, size_x, size_y):
    """Return a callable(x, y) -> surface_z for the terrain heightfield.

    Returns None if no heightfield is present.
    Must be called AFTER model loading and terrain.populate().
    """
    import mujoco

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
    if hfield_id < 0:
        return None

    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    elev_max = model.hfield_size[hfield_id][2]

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
    ground_z = model.geom_pos[ground_id][2] if ground_id >= 0 else -0.5

    hdata = model.hfield_data[:nrow * ncol].reshape(nrow, ncol).copy()

    def _surface_z(x, y):
        col = (x + size_x) / (2 * size_x) * (ncol - 1)
        row = (y + size_y) / (2 * size_y) * (nrow - 1)
        col = np.clip(col, 0, ncol - 1)
        row = np.clip(row, 0, nrow - 1)
        c0, c1 = int(col), min(int(col) + 1, ncol - 1)
        r0, r1 = int(row), min(int(row) + 1, nrow - 1)
        fc, fr = col - c0, row - r0
        h = (hdata[r0, c0] * (1 - fr) * (1 - fc)
             + hdata[r1, c0] * fr * (1 - fc)
             + hdata[r0, c1] * (1 - fr) * fc
             + hdata[r1, c1] * fr * fc)
        return ground_z + float(h) * elev_max

    return _surface_z


# ======================================================================
# Zone boundary strips
# ======================================================================

def inject_zone_boundary_strips(root: ET.Element, boundaries: list,
                                 width: float = 0.08, height: float = 0.003):
    """Add subtle thin strips at terrain zone boundaries."""
    worldbody = root.find('.//worldbody')
    if worldbody is None:
        return
    for i, bx in enumerate(boundaries):
        strip = ET.SubElement(worldbody, 'geom')
        strip.set('name', f'zone_boundary_{i}')
        strip.set('type', 'box')
        strip.set('pos', f'{bx:.2f} 0 -0.488')
        strip.set('size', f'{width} 10.0 {height}')
        strip.set('rgba', '0.4 0.4 0.4 0.5')
        strip.set('contype', '0')
        strip.set('conaffinity', '0')


# ======================================================================
# Heightfield texture painting
# ======================================================================

def paint_terrain_zones(model, zones, nrow, ncol, size_x):
    """Paint natural-looking per-zone textures into heightfield texture data.

    Each zone gets multi-scale noise to emulate real materials:
      - 'flat':   clean light grey/white concrete look
      - 'grass':  green with blade-like streaks
      - 'gravel': brown/grey with coarse grain

    Args:
        model: MjModel (after loading)
        zones: list of (x_start, x_end, rgb, style) tuples.
               rgb as (r, g, b) floats 0-1.
               style: 'flat', 'grass', or 'gravel' (default: 'flat')
        nrow, ncol: heightfield grid dimensions
        size_x: heightfield half-extent in x
    """
    import mujoco as mj
    from scipy.ndimage import uniform_filter

    tex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_TEXTURE, 'terrain_tex')
    if tex_id < 0:
        return

    tex_h = model.tex_height[tex_id]
    tex_w = model.tex_width[tex_id]
    tex_adr = model.tex_adr[tex_id]
    nc = model.tex_nchannel[tex_id]

    x_coords = np.linspace(-size_x, size_x, tex_w)
    rng = np.random.default_rng(123)

    base_rgb = np.array([0.88, 0.86, 0.82])
    tex_float = np.tile(base_rgb, (tex_h, tex_w, 1))

    blend_w = 2.0

    for zone_def in zones:
        if len(zone_def) == 4:
            x_start, x_end, zone_rgb, style = zone_def
        else:
            x_start, x_end, zone_rgb = zone_def
            style = 'flat'
        zone_rgb = np.array(zone_rgb)

        if style == 'grass':
            fine = rng.uniform(-0.12, 0.12, (tex_h, tex_w))
            coarse = rng.standard_normal((tex_h, tex_w))
            coarse = uniform_filter(coarse, size=12)
            coarse = coarse / max(coarse.max() - coarse.min(), 1e-6) * 0.15
            streak = rng.uniform(-0.06, 0.06, (1, tex_w))
            noise = fine + coarse + np.tile(streak, (tex_h, 1))
        elif style == 'gravel':
            fine = rng.uniform(-0.08, 0.08, (tex_h, tex_w))
            coarse = rng.standard_normal((tex_h, tex_w))
            coarse = uniform_filter(coarse, size=4)
            coarse = coarse / max(coarse.max() - coarse.min(), 1e-6) * 0.22
            noise = fine + coarse
        else:
            noise = rng.uniform(-0.04, 0.04, (tex_h, tex_w))

        for j in range(tex_w):
            xj = x_coords[j]
            if xj < x_start - blend_w or xj >= x_end + blend_w:
                continue
            if xj < x_start:
                w = 0.5 * (1 + np.cos(np.pi * (x_start - xj) / blend_w))
            elif xj >= x_end:
                w = 0.5 * (1 + np.cos(np.pi * (xj - x_end) / blend_w))
            else:
                w = 1.0
            for c in range(3):
                tex_float[:, j, c] = (
                    (1 - w) * tex_float[:, j, c]
                    + w * np.clip(zone_rgb[c] + noise[:, j] * zone_rgb[c],
                                  0.0, 1.0)
                )

    tex_u8 = np.clip(tex_float * 255, 0, 255).astype(np.uint8)
    model.tex_data[tex_adr:tex_adr + tex_h * tex_w * nc] = tex_u8.ravel()
