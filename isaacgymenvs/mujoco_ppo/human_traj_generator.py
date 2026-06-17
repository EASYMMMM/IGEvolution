from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class HumanTrajectoryConfig:
    dt: float
    episode_duration: float
    speed_mean: float = 1.0
    speed_jitter: float = 0.10
    turn_speed_max: float = 0.30
    forced_stand_duration: float = 0.0
    segment_duration_range: Tuple[float, float] = (2.0, 6.0)
    stand_probability: float = 0.10
    turn_probability: float = 0.30
    height: float = 1.0


class SimpleHumanTrajectoryGenerator:
    """Single-env NumPy version of the IsaacGym human trajectory idea.

    The generator produces a hidden point trajectory for the human pelvis/back
    reference. The policy does not observe this point directly; the MuJoCo env
    converts it into task commands, tracking reward, and virtual load-cell force.
    """

    def __init__(self, config: HumanTrajectoryConfig):
        self.cfg = config
        self.num_steps = max(int(np.ceil(config.episode_duration / config.dt)) + 2, 3)
        self.positions = np.zeros((self.num_steps, 3), dtype=np.float32)
        self.velocities = np.zeros((self.num_steps, 3), dtype=np.float32)
        self.yaw = np.zeros(self.num_steps, dtype=np.float32)
        self.speed = np.zeros(self.num_steps, dtype=np.float32)
        self.omega = np.zeros(self.num_steps, dtype=np.float32)

    def reset(self, init_pos: np.ndarray, init_yaw: float, rng=np.random):
        init_pos = np.asarray(init_pos, dtype=np.float32).copy()
        init_pos[2] = float(self.cfg.height)

        speed, omega = self._sample_speed_omega(rng)
        self.speed[:] = speed
        self.omega[:] = omega

        self.yaw[:] = init_yaw + np.cumsum(self.omega) * float(self.cfg.dt)
        dx = self.speed * np.cos(self.yaw) * float(self.cfg.dt)
        dy = self.speed * np.sin(self.yaw) * float(self.cfg.dt)

        self.positions[:, 0] = init_pos[0] + np.cumsum(dx) - dx[0]
        self.positions[:, 1] = init_pos[1] + np.cumsum(dy) - dy[0]
        self.positions[:, 2] = init_pos[2]

        self.velocities[:-1] = (self.positions[1:] - self.positions[:-1]) / float(self.cfg.dt)
        self.velocities[-1] = self.velocities[-2]

    def get_position(self, time_s: float) -> np.ndarray:
        return self.positions[self._time_to_index(time_s)].copy()

    def get_velocity(self, time_s: float) -> np.ndarray:
        return self.velocities[self._time_to_index(time_s)].copy()

    def get_yaw(self, time_s: float) -> float:
        return float(self.yaw[self._time_to_index(time_s)])

    def get_observation_points(self, current_time: float, future_times: Iterable[float]) -> np.ndarray:
        return np.stack(
            [self.get_position(current_time + float(offset)) for offset in future_times],
            axis=0,
        ).astype(np.float32)

    def _time_to_index(self, time_s: float) -> int:
        idx = int(np.clip(round(float(time_s) / float(self.cfg.dt)), 0, self.num_steps - 1))
        return idx

    def _sample_speed_omega(self, rng) -> Tuple[np.ndarray, np.ndarray]:
        speed = np.zeros(self.num_steps, dtype=np.float32)
        omega = np.zeros(self.num_steps, dtype=np.float32)
        dt = float(self.cfg.dt)

        step = 0
        forced_steps = int(np.clip(round(self.cfg.forced_stand_duration / dt), 0, self.num_steps))
        if forced_steps > 0:
            step = forced_steps

        min_dur, max_dur = self.cfg.segment_duration_range
        while step < self.num_steps:
            seg_duration = float(rng.uniform(min_dur, max_dur))
            seg_steps = max(int(round(seg_duration / dt)), 1)
            end = min(step + seg_steps, self.num_steps)

            r = float(rng.uniform())
            if r < self.cfg.stand_probability:
                seg_speed = 0.0
                seg_omega = 0.0
            else:
                seg_speed = float(
                    self.cfg.speed_mean
                    + rng.uniform(-self.cfg.speed_jitter, self.cfg.speed_jitter)
                )
                turn_r = float(rng.uniform())
                if turn_r < self.cfg.turn_probability:
                    sign = -1.0 if float(rng.uniform()) < 0.5 else 1.0
                    low = 0.35 * float(self.cfg.turn_speed_max)
                    high = float(self.cfg.turn_speed_max)
                    seg_omega = sign * float(rng.uniform(low, high))
                else:
                    seg_omega = 0.0

            speed[step:end] = max(seg_speed, 0.0)
            omega[step:end] = seg_omega
            step = end

        if forced_steps > 0:
            speed[:forced_steps] = 0.0
            omega[:forced_steps] = 0.0

        speed = self._smooth(speed, window=41)
        omega = self._smooth(omega, window=41)
        if forced_steps > 0:
            speed[:forced_steps] = 0.0
            omega[:forced_steps] = 0.0
        return speed.astype(np.float32), omega.astype(np.float32)

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        window = min(int(window), int(values.size))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(values, kernel, mode="same").astype(np.float32)
