"""
VideoRecorder: offscreen MuJoCo rendering to MP4.

Streaming mode — frames written directly to ffmpeg via imageio,
no RAM buffering. Typical usage:

    rec = VideoRecorder(model, data)
    rec.start('videos/test.mp4')
    for _ in range(N):
        mujoco.mj_step(model, data)
        rec.capture('overview')   # render + write one frame
    rec.stop()
    rec.close()
"""

from pathlib import Path
from datetime import datetime

import mujoco
import imageio


class VideoRecorder:
    """Offscreen MuJoCo renderer → MP4 writer (streaming)."""

    def __init__(self, model, data, width=1920, height=1080, fps=60):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fps = fps
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = None
        self._path = None
        self._frame_count = 0

    # -- recording lifecycle --------------------------------------------------

    def start(self, output_path: str):
        """Open MP4 writer. Creates parent directories if needed."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(p)
        self._writer = imageio.get_writer(
            self._path, fps=self.fps, codec='libx264',
            pixelformat='yuv420p',
            quality=8, macro_block_size=1,
        )
        self._frame_count = 0
        print(f"Recording → {self._path}  ({self.width}x{self.height} @ {self.fps}fps)")

    def capture(self, camera_name: str):
        """Render one frame from *camera_name* and write to MP4."""
        if self._writer is None:
            raise RuntimeError("Call start() before capture().")
        self.renderer.update_scene(self.data, camera=camera_name)
        frame = self.renderer.render()
        self._writer.append_data(frame)
        self._frame_count += 1

    def stop(self):
        """Flush and close the MP4 writer."""
        if self._writer is not None:
            self._writer.close()
            duration = self._frame_count / max(self.fps, 1)
            print(f"Saved {self._path}  ({self._frame_count} frames, {duration:.1f}s)")
            self._writer = None
            self._frame_count = 0

    def close(self):
        """Release the MuJoCo renderer."""
        self.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # -- helpers ---------------------------------------------------------------

    @property
    def steps_per_frame(self) -> int:
        """Physics steps between rendered frames (for 60fps at dt=0.001 → 17)."""
        return max(1, round(1.0 / (self.fps * self.model.opt.timestep)))

    @staticmethod
    def make_path(scenario: str, camera: str, base_dir: str = 'videos') -> str:
        """Generate timestamped output path: videos/{scenario}_{camera}_{ts}.mp4"""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return str(Path(base_dir) / f'{scenario}_{camera}_{ts}.mp4')
