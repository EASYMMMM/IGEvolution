"""Physics-rate sensor history, with fractional-millisecond observation delay."""

from collections import deque

import numpy as np

from mujoco_ppo.srl_mujoco_v2_env import quat_to_euler_xyz


DELAY_QUANTILE_PROBABILITIES = (0.0, 0.5, 0.95, 0.99, 1.0)
DEFAULT_COMMON_MS_QUANTILES = (0.0, 2.2, 4.5, 5.5, 6.0)
DEFAULT_SKEW_MS_QUANTILES = (0.0, 0.35, 0.9, 1.4, 1.6)


def validate_delay_quantiles(value):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError) as error:
        raise ValueError("Delay quantiles require five numbers in ms") from error
    if (len(result) != 5 or not np.isfinite(result).all() or result[0] < 0
            or any(left > right for left, right in zip(result, result[1:]))):
        raise ValueError("Delay quantiles must be five finite, nonnegative, nondecreasing numbers")
    return result


def validate_delay_range(value):
    try:
        low, high = map(float, value)
    except (TypeError, ValueError) as error:
        raise ValueError("Observation delay range requires two numbers in ms") from error
    if not np.isfinite([low, high]).all() or not 0.0 <= low <= high:
        raise ValueError("Observation delay range must satisfy 0 <= low <= high (ms)")
    return low, high


def quaternion_slerp(first, second, fraction):
    """Shortest-path interpolation of MuJoCo wxyz quaternions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second, dot = -second, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = first + fraction * (second - first)
    else:
        angle = np.arccos(dot)
        result = (np.sin((1.0 - fraction) * angle) * first
                  + np.sin(fraction * angle) * second) / np.sin(angle)
    return result / np.linalg.norm(result)


class SensorObservationDelay:
    """Legacy independent delays, or correlated ages with small signed skew.

    History holds clean sensors, before observation noise. Commands, previous
    actions, phase and privileged labels are intentionally outside this buffer.
    """

    def __init__(self, enabled=False, imu_ms_range=(0.0, 5.0),
                 joint_ms_range=(0.0, 0.0), *, mode="independent",
                 common_ms_quantiles=DEFAULT_COMMON_MS_QUANTILES,
                 skew_ms_quantiles=DEFAULT_SKEW_MS_QUANTILES,
                 resample_per_observation=False):
        self.enabled = bool(enabled)
        self.imu_range = validate_delay_range(imu_ms_range)
        self.joint_range = validate_delay_range(joint_ms_range)
        self.mode = str(mode)
        if self.mode not in ("independent", "correlated"):
            raise ValueError("Observation delay mode must be independent or correlated")
        self.common_quantiles = validate_delay_quantiles(common_ms_quantiles)
        self.skew_quantiles = validate_delay_quantiles(skew_ms_quantiles)
        self.resample_per_observation = bool(resample_per_observation)
        if self.mode == "independent" and self.resample_per_observation:
            raise ValueError("Per-observation resampling requires correlated mode; legacy delays stay per episode")
        self.rng = np.random.default_rng()
        self.history = deque()
        self.imu_ms = self.joint_ms = 0.0
        self._last_observation_time = None

    @property
    def max_delay_ms(self):
        if self.mode == "correlated":
            common_max = self.common_quantiles[-1]
            return common_max + min(self.skew_quantiles[-1], 2 * common_max) / 2
        return max(self.imu_ms, self.joint_ms)

    @property
    def active(self):
        maximum = (self.max_delay_ms if self.mode == "correlated" and self.resample_per_observation
                   else max(self.imu_ms, self.joint_ms))
        return self.enabled and maximum > 0.0

    def begin_episode(self, seed=None):
        # Never consume the environment RNG: enabling zero delay must not shift DR.
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454E53]))
        self.history.clear()
        self._last_observation_time = None
        self.imu_ms, self.joint_ms = self._sample_delays() if self.enabled else (0.0, 0.0)

    def _sample_quantiles(self, values):
        if values[0] == values[-1]:
            return values[0]
        return float(np.interp(self.rng.random(), DELAY_QUANTILE_PROBABILITIES, values))

    def _sample_delays(self):
        if self.mode == "independent":
            return self._sample(self.imu_range), self._sample(self.joint_range)
        if self.common_quantiles[-1] == 0.0:
            return 0.0, 0.0
        common = self._sample_quantiles(self.common_quantiles)
        # Bound the skew near reset/zero age so neither sensor reads the future.
        skew = min(self._sample_quantiles(self.skew_quantiles), 2 * common)
        if skew > 0.0 and self.rng.integers(2) == 0:
            skew = -skew
        return common - skew / 2, common + skew / 2

    def _select_observation_time(self, timestamp):
        if self.mode != "correlated" or not self.resample_per_observation:
            return
        timestamp = float(timestamp)
        if self._last_observation_time is not None and timestamp < self._last_observation_time:
            raise ValueError("Observation times must be monotonic; begin a new episode on reset")
        if self._last_observation_time is not None and timestamp != self._last_observation_time:
            self.imu_ms, self.joint_ms = self._sample_delays()
        self._last_observation_time = timestamp

    def _sample(self, bounds):
        return bounds[0] if bounds[0] == bounds[1] else float(self.rng.uniform(*bounds))

    def capture(self, timestamp, quaternion, angular_velocity, joint_pos, joint_vel):
        if not self.active:
            return
        timestamp = float(timestamp)
        if self.history and timestamp < self.history[-1][0]:
            raise ValueError("Sensor timestamps must be monotonic; clear history on reset")
        sample = (timestamp, *[np.asarray(value, dtype=np.float64).copy() for value in
                              (quaternion, angular_velocity, joint_pos, joint_vel)])
        if self.history and timestamp == self.history[-1][0]:
            self.history[-1] = sample
        else:
            self.history.append(sample)
        # A later frame can request a larger age; retain the configured envelope,
        # not only the delays sampled for the preceding frame.
        cutoff = timestamp - self.max_delay_ms * 0.001
        while len(self.history) > 2 and self.history[1][0] <= cutoff:
            self.history.popleft()

    def _bracket(self, timestamp):
        if not self.history:
            raise RuntimeError("Capture reset sensors before requesting delayed observations")
        if timestamp <= self.history[0][0]:
            return self.history[0], self.history[0], 0.0
        for index in range(len(self.history) - 1, 0, -1):
            before, after = self.history[index - 1], self.history[index]
            if timestamp >= before[0]:
                blend = np.clip((timestamp - before[0]) / (after[0] - before[0]), 0.0, 1.0)
                return before, after, float(blend)
        raise RuntimeError("Invalid sensor history")

    def apply(self, clean_frame, timestamp, target_yaw, default_joint_pos,
              dof_vel_obs_scale, clip_obs):
        if not self.active:
            return clean_frame
        self._select_observation_time(timestamp)
        if max(self.imu_ms, self.joint_ms) == 0.0:
            return clean_frame
        frame = clean_frame.copy()
        if self.imu_ms > 0.0:
            before, after, blend = self._bracket(timestamp - self.imu_ms * 0.001)
            quat = quaternion_slerp(before[1], after[1], blend)
            angles = quat_to_euler_xyz(quat)
            yaw_error = target_yaw - angles[0]
            frame[4:7] = before[2] + blend * (after[2] - before[2])
            frame[7:10] = [np.arctan2(np.sin(yaw_error), np.cos(yaw_error)),
                           -angles[1], -angles[2]]
        if self.joint_ms > 0.0:
            before, after, blend = self._bracket(timestamp - self.joint_ms * 0.001)
            frame[10:16] = before[3] + blend * (after[3] - before[3]) - default_joint_pos
            frame[16:22] = (before[4] + blend * (after[4] - before[4])) * dof_vel_obs_scale
        return np.clip(frame, -clip_obs, clip_obs).astype(np.float32)
