"""Physics-rate sensor history, with fractional-millisecond observation delay."""

from collections import deque

import numpy as np

from mujoco_ppo.srl_mujoco_v2_env import quat_to_euler_xyz


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
    """Independent IMU/joint delays, sampled once per episode, not per frame.

    History holds clean sensors, before observation noise. Commands, previous
    actions, phase and privileged labels are intentionally outside this buffer.
    """

    def __init__(self, enabled=False, imu_ms_range=(0.0, 5.0),
                 joint_ms_range=(0.0, 0.0)):
        self.enabled = bool(enabled)
        self.imu_range = validate_delay_range(imu_ms_range)
        self.joint_range = validate_delay_range(joint_ms_range)
        self.rng = np.random.default_rng()
        self.history = deque()
        self.imu_ms = self.joint_ms = 0.0

    @property
    def active(self):
        return self.enabled and max(self.imu_ms, self.joint_ms) > 0.0

    def begin_episode(self, seed=None):
        # Never consume the environment RNG: enabling zero delay must not shift DR.
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454E53]))
        self.history.clear()
        self.imu_ms = self._sample(self.imu_range) if self.enabled else 0.0
        self.joint_ms = self._sample(self.joint_range) if self.enabled else 0.0

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
        cutoff = timestamp - max(self.imu_ms, self.joint_ms) * 0.001
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
