"""
Terrain generators for MuJoCo heightfield-based ground surfaces.

Each terrain class provides:
  - inject_xml(root): Replace flat ground plane with hfield asset+geom in XML tree
  - populate(model): Fill model.hfield_data after model loading

Usage:
    terrain = SlopeTerrain(angle_deg=5)
    sim = RobotSimulation(terrain=terrain)

Available terrains:
    FlatTerrain    — no-op, keeps existing ground plane
    SlopeTerrain   — inclined plane along x or y
    WavyTerrain    — sinusoidal hills
    StepTerrain    — staircase (discrete elevation steps)
    RoughTerrain   — Gaussian-filtered random noise
"""

import numpy as np
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


class Terrain(ABC):
    """Base class for terrain generators."""

    def __init__(self, nrow=200, ncol=200, size_x=100.0, size_y=100.0,
                 elevation_max=0.5, base_depth=0.1):
        self.nrow = nrow
        self.ncol = ncol
        self.size_x = size_x      # half-extent in x (meters)
        self.size_y = size_y      # half-extent in y (meters)
        self.elevation_max = elevation_max  # max height scale (meters)
        self.base_depth = base_depth

    def inject_xml(self, root: ET.Element):
        """Replace ground plane with hfield asset+geom in XML tree."""
        # Remove existing ground geom
        worldbody = root.find('.//worldbody')
        if worldbody is None:
            return
        ground = worldbody.find('geom[@name="ground"]')
        if ground is not None:
            # Preserve contact attributes
            worldbody.remove(ground)

        # Add hfield to asset section
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')

        hfield = ET.SubElement(asset, 'hfield')
        hfield.set('name', 'terrain')
        hfield.set('nrow', str(self.nrow))
        hfield.set('ncol', str(self.ncol))
        hfield.set('size', f'{self.size_x} {self.size_y} '
                   f'{self.elevation_max} {self.base_depth}')

        # Add hfield geom to worldbody
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('name', 'ground')
        geom.set('type', 'hfield')
        geom.set('hfield', 'terrain')
        geom.set('pos', '0 0 -0.5')
        geom.set('rgba', '0.92 0.9 0.86 1.0')
        geom.set('friction', '0.3 0.02 0.01')
        geom.set('condim', '4')
        geom.set('solref', '0.02 1')
        geom.set('solimp', '0.9 0.95 0.01')
        geom.set('contype', '1')
        geom.set('conaffinity', '1')

        # Visual-only far ground avoids black background showing beyond the
        # heightfield bounds.  Made very large so the edge is never visible.
        # fog_extent in inject_video_visuals overrides stat.extent so this
        # size does not inflate fog distances.
        far = ET.SubElement(worldbody, 'geom')
        far.set('name', 'ground_far')
        far.set('type', 'plane')
        far.set('size', f'{self.size_x * 5} {self.size_y * 5} 0.1')
        far.set('pos', '0 0 -0.505')
        far.set('rgba', '0.90 0.89 0.86 1.0')
        far.set('contype', '0')
        far.set('conaffinity', '0')

    @abstractmethod
    def generate_heights(self) -> np.ndarray:
        """Generate height array (nrow x ncol), values in [0, 1]."""
        ...

    def populate(self, model):
        """Fill model.hfield_data with centered heights.

        Centers the terrain so the surface at the origin (grid center)
        sits at z = -0.5, same as the original flat ground plane.
        This prevents the robot from starting embedded in the terrain.

        Adjusts model.hfield_size (elevation_max) and model.geom_pos
        (ground geom z) after writing the height data.
        """
        import mujoco

        heights = self.generate_heights()  # [0, 1] normalized
        h_phys = heights * self.elevation_max  # physical meters

        # Center: make height at grid center (origin) = 0
        cr, cc = self.nrow // 2, self.ncol // 2
        h_center = h_phys[cr, cc]
        h_phys -= h_center  # origin now at 0, some values may be negative

        # Shift to non-negative (hfield data must be in [0, 1])
        h_min = float(h_phys.min())
        h_phys -= h_min  # now min = 0

        # Recompute elevation_max and normalize
        new_elev_max = max(float(h_phys.max()), 1e-6)
        h_norm = np.clip(h_phys / new_elev_max, 0.0, 1.0).astype(np.float32)

        # Write height data
        model.hfield_data[:] = h_norm.ravel()

        # Update hfield elevation_max in model
        hfield_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_HFIELD, 'terrain')
        if hfield_id >= 0:
            model.hfield_size[hfield_id][2] = new_elev_max

        # Adjust ground geom z so surface at origin = -0.5
        # Surface at center = geom_z + h_norm[cr,cc] * new_elev_max
        #                   = geom_z + (-h_min)
        # Want: geom_z + (-h_min) = -0.5  =>  geom_z = -0.5 + h_min
        ground_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
        if ground_id >= 0:
            model.geom_pos[ground_id][2] = -0.5 + h_min


class FlatTerrain(Terrain):
    """No-op terrain — keeps existing ground plane."""

    def inject_xml(self, root: ET.Element):
        pass  # keep ground plane as-is

    def generate_heights(self) -> np.ndarray:
        return np.zeros((self.nrow, self.ncol))

    def populate(self, model):
        pass  # no hfield to fill


class SlopeTerrain(Terrain):
    """Inclined plane along x or y direction, spanning the full grid."""

    def __init__(self, angle_deg=5.0, direction='x', **kwargs):
        # Auto-compute elevation_max from actual rise across full extent
        half = kwargs.get('size_x', 100.0) if direction == 'x' else kwargs.get('size_y', 100.0)
        rise = 2 * half * np.tan(np.radians(angle_deg))
        kwargs.setdefault('elevation_max', max(rise, 0.01))
        super().__init__(**kwargs)
        self.angle_deg = angle_deg
        self.direction = direction

    def generate_heights(self) -> np.ndarray:
        # Linear gradient from 0 to 1 across the full extent
        if self.direction == 'x':
            grad = np.linspace(0.0, 1.0, self.ncol)
            heights = np.tile(grad, (self.nrow, 1))
        else:
            grad = np.linspace(0.0, 1.0, self.nrow)
            heights = np.tile(grad[:, None], (1, self.ncol))
        return heights


class WavyTerrain(Terrain):
    """Sinusoidal hills along x direction, spanning the full grid."""

    def __init__(self, amplitude=0.05, wavelength=2.0, **kwargs):
        kwargs.setdefault('elevation_max', max(amplitude, 0.01))
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.wavelength = wavelength

    def generate_heights(self) -> np.ndarray:
        x = np.linspace(-self.size_x, self.size_x, self.ncol)
        # sin -> [0, 1] range: (1 + sin) / 2
        h1d = (1.0 + np.sin(2 * np.pi * x / self.wavelength)) / 2.0
        heights = np.tile(h1d, (self.nrow, 1))
        return heights


class StepTerrain(Terrain):
    """Staircase terrain with discrete elevation steps along x, spanning the full grid."""

    def __init__(self, step_height=0.03, num_steps=5, **kwargs):
        total_rise = num_steps * step_height
        kwargs.setdefault('elevation_max', max(total_rise, 0.01))
        super().__init__(**kwargs)
        self.step_height = step_height
        self.num_steps = num_steps

    def generate_heights(self) -> np.ndarray:
        x = np.linspace(-self.size_x, self.size_x, self.ncol)
        # Divide the full x-extent into num_steps equal bands
        # band 0 (leftmost) = lowest, band num_steps-1 = highest
        step_width = 2.0 * self.size_x / self.num_steps
        h1d = np.zeros(self.ncol)
        for i in range(self.num_steps):
            threshold = -self.size_x + (i + 1) * step_width
            h1d[x >= threshold] = (i + 1) / self.num_steps
        heights = np.tile(h1d, (self.nrow, 1))
        return heights


class RoughTerrain(Terrain):
    """Random bumpy terrain with Gaussian filtering, spanning the full grid."""

    def __init__(self, roughness=0.02, seed=42, **kwargs):
        # roughness is the peak-to-peak height range in meters
        kwargs.setdefault('elevation_max', max(roughness * 6, 0.01))
        super().__init__(**kwargs)
        self.roughness = roughness
        self.seed = seed

    def generate_heights(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        raw = rng.standard_normal((self.nrow, self.ncol))
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(raw, size=5)
        smoothed = uniform_filter(smoothed, size=5)
        # Normalize to [0, 1]: shift min→0, then scale so that
        # the physical range (after * elevation_max) ≈ roughness * 6σ
        heights = smoothed - smoothed.min()
        h_range = heights.max()
        if h_range > 0:
            heights = heights / h_range
        return heights


class MultiZoneTerrain(Terrain):
    """Multiple terrain zones along x-axis with different roughness levels.

    Each zone is defined by (x_start, x_end, roughness) in world coordinates.
    Areas not covered by any zone remain flat.

    Usage:
        zones = [
            (-5, 5, 0.0),     # smooth flat
            (5, 15, 0.02),    # mild rough
            (15, 30, 0.06),   # rough
        ]
        terrain = MultiZoneTerrain(zones=zones, seed=42)
    """

    def __init__(self, zones, seed=42, **kwargs):
        max_rough = max(r for _, _, r in zones) if zones else 0.01
        kwargs.setdefault('elevation_max', max(max_rough * 6, 0.01))
        super().__init__(**kwargs)
        self.zones = zones
        self.seed = seed

    def generate_heights(self) -> np.ndarray:
        """Generate continuous noise across full grid, scaled per zone.

        Uses a single noise field so there are no seams at zone boundaries.
        A smooth blending region (cosine fade) prevents abrupt height jumps.
        """
        from scipy.ndimage import uniform_filter
        rng = np.random.default_rng(self.seed)

        # Single continuous noise field
        raw = rng.standard_normal((self.nrow, self.ncol))
        smoothed = uniform_filter(raw, size=5)
        smoothed = uniform_filter(smoothed, size=5)
        # Normalize to [0, 1]
        smoothed -= smoothed.min()
        if smoothed.max() > 0:
            smoothed /= smoothed.max()

        # Build per-column roughness scale with smooth blending
        x = np.linspace(-self.size_x, self.size_x, self.ncol)
        blend_width = 2.0  # meters of cosine crossfade at zone boundaries
        scale = np.zeros(self.ncol)
        for x_start, x_end, roughness in self.zones:
            if roughness <= 0:
                continue
            for j in range(self.ncol):
                if x[j] < x_start - blend_width or x[j] >= x_end + blend_width:
                    continue
                # Smooth ramp in
                if x[j] < x_start:
                    w = 0.5 * (1 + np.cos(np.pi * (x_start - x[j]) / blend_width))
                elif x[j] >= x_end:
                    w = 0.5 * (1 + np.cos(np.pi * (x[j] - x_end) / blend_width))
                else:
                    w = 1.0
                target = roughness * 6 / self.elevation_max
                scale[j] = max(scale[j], w * target)

        heights = smoothed * scale[np.newaxis, :]
        return np.clip(heights, 0.0, 1.0)

    def inject_xml(self, root):
        """Inject heightfield + a texture for zone colouring."""
        super().inject_xml(root)

        # Add a texture + material for the heightfield so we can paint zones
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')

        tex = ET.SubElement(asset, 'texture')
        tex.set('name', 'terrain_tex')
        tex.set('type', '2d')
        tex.set('builtin', 'flat')
        tex.set('rgb1', '0.88 0.86 0.82')
        tex.set('width', str(self.ncol))
        tex.set('height', str(self.nrow))

        mat = ET.SubElement(asset, 'material')
        mat.set('name', 'terrain_mat')
        mat.set('texture', 'terrain_tex')
        mat.set('texrepeat', '1 1')
        mat.set('reflectance', '0.05')

        # Apply material to ground geom
        ground = root.find('.//geom[@name="ground"]')
        if ground is not None:
            ground.set('material', 'terrain_mat')


# ======================================================================
# Visualization
# ======================================================================

def visualize_terrain(terrain, slice_positions=None, save_path=None):
    """Visualize terrain height distribution: 3D surface + XY cross-sections.

    Args:
        terrain:         Terrain instance (non-Flat).
        slice_positions: Dict with 'x' and 'y' lists of positions (meters)
                         for cross-section cuts. Default: origin + a few offsets.
        save_path:       If given, save figure to this path.

    Produces 4 subplots:
        1. 3D surface view
        2. Top-down heatmap with grid slice lines
        3. Height profiles along X at various Y positions
        4. Height profiles along Y at various X positions
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    heights_norm = terrain.generate_heights()
    heights_m = heights_norm * terrain.elevation_max  # convert to meters

    # Center so origin height = 0 (matches what populate() does in simulation)
    cr, cc = terrain.nrow // 2, terrain.ncol // 2
    heights_m -= heights_m[cr, cc]

    x = np.linspace(-terrain.size_x, terrain.size_x, terrain.ncol)
    y = np.linspace(-terrain.size_y, terrain.size_y, terrain.nrow)

    if slice_positions is None:
        slice_positions = {
            'x': [0.0, -3.0, 3.0],   # slice along x-axis at these y values
            'y': [0.0, -3.0, 3.0],   # slice along y-axis at these x values
        }

    # Colors for slice lines
    colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{type(terrain).__name__} — Height Distribution', fontsize=15)

    # --- 1. 3D surface ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    X, Y = np.meshgrid(x, y)
    stride = max(1, terrain.nrow // 50)
    ax1.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                     heights_m[::stride, ::stride],
                     cmap='terrain', alpha=0.85, edgecolor='none')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Height (m)')
    ax1.set_title('3D Surface')

    # --- 2. Top-down heatmap with slice grid lines ---
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(heights_m, extent=[-terrain.size_x, terrain.size_x,
                                        -terrain.size_y, terrain.size_y],
                    origin='lower', cmap='terrain', aspect='equal')
    fig.colorbar(im, ax=ax2, label='Height (m)', shrink=0.8)
    # Draw slice lines
    for i, yp in enumerate(slice_positions['x']):
        c = colors[i % len(colors)]
        ax2.axhline(yp, color=c, linestyle='--', linewidth=1.2,
                    label=f'Y={yp:+.1f}m')
    for i, xp in enumerate(slice_positions['y']):
        c = colors[i % len(colors)]
        ax2.axvline(xp, color=c, linestyle=':', linewidth=1.2,
                    label=f'X={xp:+.1f}m')
    ax2.plot(0, 0, 'w+', markersize=12, markeredgewidth=2)  # origin marker
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Height Map + Slice Grid')
    ax2.legend(loc='upper right', fontsize=7)

    # --- 3. Height profiles along X (at fixed Y positions) ---
    ax3 = fig.add_subplot(2, 2, 3)
    for i, yp in enumerate(slice_positions['x']):
        row_idx = int(np.clip((yp + terrain.size_y) / (2 * terrain.size_y)
                              * (terrain.nrow - 1), 0, terrain.nrow - 1))
        c = colors[i % len(colors)]
        ax3.plot(x, heights_m[row_idx, :], color=c, linewidth=1.5,
                 label=f'Y={yp:+.1f}m')
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='Origin')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Cross-section along X  (origin = 0)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 4. Height profiles along Y (at fixed X positions) ---
    ax4 = fig.add_subplot(2, 2, 4)
    for i, xp in enumerate(slice_positions['y']):
        col_idx = int(np.clip((xp + terrain.size_x) / (2 * terrain.size_x)
                              * (terrain.ncol - 1), 0, terrain.ncol - 1))
        c = colors[i % len(colors)]
        ax4.plot(y, heights_m[:, col_idx], color=c, linewidth=1.5,
                 label=f'X={xp:+.1f}m')
    ax4.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='Origin')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Cross-section along Y  (origin = 0)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150)
        print(f"Saved terrain visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize terrain heightfields")
    parser.add_argument('--terrain', type=str, default='wavy',
                        choices=['slope', 'wavy', 'steps', 'rough'],
                        help='Terrain type to visualize')
    parser.add_argument('--slope-angle', type=float, default=5.0)
    parser.add_argument('--amplitude', type=float, default=0.05)
    parser.add_argument('--wavelength', type=float, default=2.0)
    parser.add_argument('--step-height', type=float, default=0.03)
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--roughness', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to path (e.g. terrain_viz.png)')
    parser.add_argument('--slices-x', type=float, nargs='+',
                        default=[0.0, -3.0, 3.0],
                        help='Y positions for X-direction slices')
    parser.add_argument('--slices-y', type=float, nargs='+',
                        default=[0.0, -3.0, 3.0],
                        help='X positions for Y-direction slices')
    args = parser.parse_args()

    if args.terrain == 'slope':
        t = SlopeTerrain(angle_deg=args.slope_angle)
    elif args.terrain == 'wavy':
        t = WavyTerrain(amplitude=args.amplitude, wavelength=args.wavelength)
    elif args.terrain == 'steps':
        t = StepTerrain(step_height=args.step_height, num_steps=args.num_steps)

    elif args.terrain == 'rough':
        t = RoughTerrain(roughness=args.roughness, seed=args.seed, nrow=400, ncol=400)

    slices = {'x': args.slices_x, 'y': args.slices_y}
    visualize_terrain(t, slice_positions=slices, save_path=args.save)
