"""Video recording helpers for policy evaluation."""

from pathlib import Path
import tempfile

import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from PIL import Image


class FirstEnvFirstEpisodeRecorder(gym.Wrapper):
    """Record env 0 until its first done signal into one exact MP4 path."""

    def __init__(self, env, video_path):
        super().__init__(env)

        self.video_path = Path(video_path).expanduser().resolve()
        if self.video_path.suffix.lower() != ".mp4":
            raise ValueError(f"Video output must use the .mp4 extension: {self.video_path}")

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        if getattr(self.env, "viewer", None) is None:
            raise RuntimeError(
                "Video recording requires an Isaac Gym viewer; use headless=False."
            )

        self.video_encoder = None
        self.frame_directory = None
        self.frame_path = None
        self.recording = False
        self.recorded_frames = 0
        self.finished = False
        self.frames_per_second = self._get_frames_per_second()

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        if not self.finished and not self.recording:
            self._start_video_recorder()
        return observations

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        if self.recording:
            # VecTask.step() already renders when force_render=True. Avoid drawing
            # the same simulation state twice before taking the viewer screenshot.
            self._capture_frame(render_first=not getattr(self.env, "force_render", False))
            if self._env_zero_is_done(dones):
                self._close_video_recorder()
                self.finished = True
                print(
                    f"Recorded env 0 first episode ({self.recorded_frames} frames) "
                    f"to {self.video_path}"
                )

        return observations, rewards, dones, infos

    def _start_video_recorder(self):
        self._close_video_recorder()
        self.frame_directory = tempfile.TemporaryDirectory(
            prefix="srl_first_episode_video_"
        )
        self.frame_path = Path(self.frame_directory.name) / "viewer_frame.png"
        self.recording = True
        self.recorded_frames = 0
        self._capture_frame(render_first=True)
        print(f"Recording env 0 first episode to {self.video_path}")

    def _capture_frame(self, render_first):
        if render_first:
            self.env.render(mode="human")

        self.env.gym.write_viewer_image_to_file(
            self.env.viewer,
            str(self.frame_path),
        )
        with Image.open(self.frame_path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

        if self.video_encoder is None:
            self.video_encoder = ImageEncoder(
                str(self.video_path),
                frame.shape,
                self.frames_per_second,
                self.frames_per_second,
            )
        self.video_encoder.capture_frame(frame)
        self.recorded_frames += 1

    def _get_frames_per_second(self):
        render_fps = int(getattr(self.env, "render_fps", -1))
        if render_fps > 0:
            return render_fps

        control_dt = float(getattr(self.env, "control_dt", 0.0))
        if control_dt > 0.0:
            return max(1, int(round(1.0 / control_dt)))

        return int(self.env.metadata.get("video.frames_per_second", 24))

    @staticmethod
    def _env_zero_is_done(dones):
        try:
            env_zero_done = dones[0]
        except (IndexError, TypeError):
            env_zero_done = dones

        if hasattr(env_zero_done, "item"):
            env_zero_done = env_zero_done.item()
        return bool(env_zero_done)

    def _close_video_recorder(self):
        video_encoder = getattr(self, "video_encoder", None)
        if video_encoder is not None:
            video_encoder.close()
            self.video_encoder = None

        frame_directory = getattr(self, "frame_directory", None)
        if frame_directory is not None:
            frame_directory.cleanup()
            self.frame_directory = None
            self.frame_path = None
        self.recording = False

    def close(self):
        self._close_video_recorder()
        return self.env.close()

    def __del__(self):
        self._close_video_recorder()
