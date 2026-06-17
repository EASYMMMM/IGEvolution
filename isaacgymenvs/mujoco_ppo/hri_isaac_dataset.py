from __future__ import annotations

import bisect
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


DEFAULT_FIELDS = (
    "srl_obs",
    "srl_full_obs",
    "teacher_srl_obs",
    "raw_mu_srl",
    "action_srl_applied",
    "virtual_load_cell",
    "target_vel_x",
    "target_ang_vel_z",
    "target_pelvis_height",
    "target_yaw",
    "root_states",
    "srl_root_states",
    "dof_pos",
    "dof_vel",
    "srl_end_pos",
    "srl_end_vel",
    "srl_reward",
    "reward",
    "done",
)


@dataclass(frozen=True)
class IsaacHRIObsSpec:
    frame_stack: int = 5
    policy_frame_dim: int = 39
    full_frame_dim: int = 43
    teacher_frame_dim: int = 30
    task_dim: int = 3

    @property
    def policy_obs_dim(self) -> int:
        return self.frame_stack * self.policy_frame_dim + self.task_dim

    @property
    def full_obs_dim(self) -> int:
        return self.frame_stack * self.full_frame_dim + self.task_dim

    @property
    def teacher_obs_dim(self) -> int:
        return self.frame_stack * self.teacher_frame_dim + self.task_dim


class IsaacHRIChunkDataset:
    """Lazy reader for SRL-HRI trajectories collected from IsaacGym.

    The collector stores chunk tensors as [T, num_envs, ...]. This dataset treats
    each (time, env) pair as one sample and loads chunks on demand.
    """

    def __init__(
        self,
        data_dir: str,
        pattern: str = "srl_hri_traj_chunk_*.pt",
        fields: Sequence[str] = DEFAULT_FIELDS,
        cache_size: int = 2,
        map_location: str = "cpu",
    ):
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.pattern = pattern
        self.fields = tuple(fields)
        self.cache_size = int(cache_size)
        self.map_location = map_location

        self.chunk_paths = sorted(glob.glob(os.path.join(self.data_dir, self.pattern)))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No Isaac HRI dataset chunks found: {self.data_dir}/{self.pattern}")

        self.chunk_lengths: List[int] = []
        self.chunk_shapes: List[Tuple[int, int]] = []
        self.metadata: List[Mapping] = []
        self._cache: Dict[int, Mapping] = {}
        self._cache_order: List[int] = []

        for path in self.chunk_paths:
            chunk = self._torch_load(path)
            if "srl_obs" not in chunk:
                raise KeyError(f"{path} does not contain required key 'srl_obs'")
            srl_obs = chunk["srl_obs"]
            if not torch.is_tensor(srl_obs) or srl_obs.ndim < 3:
                raise ValueError(f"{path} has invalid srl_obs shape: {getattr(srl_obs, 'shape', None)}")
            t, n = int(srl_obs.shape[0]), int(srl_obs.shape[1])
            self.chunk_shapes.append((t, n))
            self.chunk_lengths.append(t * n)
            self.metadata.append(chunk.get("metadata", {}))

        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.total_samples = int(self.cumulative_lengths[-1])

    def __len__(self) -> int:
        return self.total_samples

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_paths)

    def summary(self) -> Dict:
        first_meta = self.metadata[0] if self.metadata else {}
        return {
            "data_dir": self.data_dir,
            "num_chunks": self.num_chunks,
            "total_samples": self.total_samples,
            "chunk_shapes": self.chunk_shapes[:5],
            "first_metadata": dict(first_meta),
        }

    def _torch_load(self, path: str) -> Mapping:
        try:
            return torch.load(path, map_location=self.map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.map_location)

    def _load_chunk(self, chunk_id: int) -> Mapping:
        if chunk_id in self._cache:
            return self._cache[chunk_id]

        chunk = self._torch_load(self.chunk_paths[chunk_id])
        self._cache[chunk_id] = chunk
        self._cache_order.append(chunk_id)

        while self.cache_size >= 0 and len(self._cache_order) > self.cache_size:
            old_id = self._cache_order.pop(0)
            self._cache.pop(old_id, None)
        return chunk

    def _locate(self, global_index: int) -> Tuple[int, int]:
        if global_index < 0 or global_index >= self.total_samples:
            raise IndexError(f"sample index out of range: {global_index}")
        chunk_id = int(bisect.bisect_right(self.cumulative_lengths, global_index))
        prev_end = 0 if chunk_id == 0 else int(self.cumulative_lengths[chunk_id - 1])
        local_flat = int(global_index - prev_end)
        return chunk_id, local_flat

    @staticmethod
    def _flatten_time_env(value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2:
            return value
        return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    def get(self, global_index: int, fields: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        fields = tuple(fields or self.fields)
        chunk_id, local_flat = self._locate(int(global_index))
        chunk = self._load_chunk(chunk_id)

        sample = {}
        for key in fields:
            if key not in chunk or not torch.is_tensor(chunk[key]):
                continue
            flat = self._flatten_time_env(chunk[key])
            sample[key] = flat[local_flat].clone()
        return sample

    def sample_batch(
        self,
        batch_size: int,
        fields: Optional[Sequence[str]] = None,
        device: Optional[Union[torch.device, str]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        fields = tuple(fields or self.fields)
        indices = torch.randint(
            low=0,
            high=self.total_samples,
            size=(int(batch_size),),
            generator=generator,
        ).tolist()

        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for out_pos, global_index in enumerate(indices):
            chunk_id, local_flat = self._locate(global_index)
            grouped.setdefault(chunk_id, []).append((out_pos, local_flat))

        out: Dict[str, List[Tuple[int, torch.Tensor]]] = {key: [] for key in fields}
        for chunk_id, positions in grouped.items():
            chunk = self._load_chunk(chunk_id)
            local_indices = torch.tensor([p[1] for p in positions], dtype=torch.long)
            out_positions = [p[0] for p in positions]

            for key in fields:
                if key not in chunk or not torch.is_tensor(chunk[key]):
                    continue
                flat = self._flatten_time_env(chunk[key])
                values = flat.index_select(0, local_indices)
                out[key].append((out_positions, values))

        batch: Dict[str, torch.Tensor] = {}
        for key, parts in out.items():
            if not parts:
                continue
            first = parts[0][1]
            merged = torch.empty(
                (batch_size, *first.shape[1:]),
                dtype=first.dtype,
                device=first.device,
            )
            for out_positions, values in parts:
                merged[torch.tensor(out_positions, dtype=torch.long)] = values
            if device is not None:
                merged = merged.to(device)
            batch[key] = merged

        return batch

    def sample_sequence(
        self,
        seq_len: int,
        fields: Optional[Sequence[str]] = None,
        device: Optional[Union[torch.device, str]] = None,
        generator: Optional[torch.Generator] = None,
        avoid_done: bool = True,
        allow_last_done: bool = True,
        max_tries: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Sample one continuous [seq_len] segment from one chunk and env.

        The returned tensors have shape [seq_len, ...]. By default, the segment
        is rejected if it crosses an episode reset according to the collected
        `done` tensor. If allow_last_done=True, a terminal flag is allowed only
        on the final frame.
        """
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        valid_chunks = [i for i, (t, _n) in enumerate(self.chunk_shapes) if t >= seq_len]
        if not valid_chunks:
            raise ValueError(f"No chunk is long enough for seq_len={seq_len}")

        last_error = None
        for _ in range(int(max_tries)):
            chunk_list_idx = int(torch.randint(0, len(valid_chunks), (1,), generator=generator).item())
            chunk_id = valid_chunks[chunk_list_idx]
            t_steps, num_envs = self.chunk_shapes[chunk_id]
            env_id = int(torch.randint(0, num_envs, (1,), generator=generator).item())
            start = int(torch.randint(0, t_steps - seq_len + 1, (1,), generator=generator).item())

            chunk = self._load_chunk(chunk_id)
            if avoid_done and "done" in chunk and torch.is_tensor(chunk["done"]):
                done_slice = chunk["done"][start : start + seq_len, env_id]
                done_check = done_slice[:-1] if allow_last_done and done_slice.numel() > 1 else done_slice
                if bool(done_check.any().item()):
                    last_error = (
                        f"sample crosses done: chunk={chunk_id}, env={env_id}, "
                        f"start={start}, seq_len={seq_len}"
                    )
                    continue

            return self.get_sequence(
                chunk_id=chunk_id,
                env_id=env_id,
                start=start,
                seq_len=seq_len,
                fields=fields,
                device=device,
            )

        raise RuntimeError(
            f"Failed to sample a valid sequence after {max_tries} tries. "
            f"Try a shorter seq_len or set avoid_done=False. Last error: {last_error}"
        )

    def get_sequence(
        self,
        chunk_id: int,
        env_id: int,
        start: int,
        seq_len: int,
        fields: Optional[Sequence[str]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a deterministic continuous segment from one chunk/env."""
        fields = tuple(fields or self.fields)
        chunk_id = int(chunk_id)
        env_id = int(env_id)
        start = int(start)
        seq_len = int(seq_len)

        if chunk_id < 0 or chunk_id >= self.num_chunks:
            raise IndexError(f"chunk_id out of range: {chunk_id}")
        t_steps, num_envs = self.chunk_shapes[chunk_id]
        if env_id < 0 or env_id >= num_envs:
            raise IndexError(f"env_id out of range: {env_id}")
        if start < 0 or start + seq_len > t_steps:
            raise IndexError(
                f"sequence range out of chunk bounds: start={start}, seq_len={seq_len}, chunk_T={t_steps}"
            )

        chunk = self._load_chunk(chunk_id)
        seq: Dict[str, torch.Tensor] = {}
        for key in fields:
            if key not in chunk or not torch.is_tensor(chunk[key]):
                continue
            value = chunk[key][start : start + seq_len, env_id].clone()
            if device is not None:
                value = value.to(device)
            seq[key] = value

        seq["sequence_info"] = {
            "chunk_id": chunk_id,
            "env_id": env_id,
            "start": start,
            "seq_len": seq_len,
            "path": self.chunk_paths[chunk_id],
        }
        return seq


class IsaacHRISequenceReplay:
    """Small helper for stepping through one sampled Isaac HRI sequence."""

    def __init__(
        self,
        dataset: IsaacHRIChunkDataset,
        seq_len: int,
        fields: Optional[Sequence[str]] = None,
        device: Optional[Union[torch.device, str]] = None,
        generator: Optional[torch.Generator] = None,
        avoid_done: bool = True,
    ):
        self.dataset = dataset
        self.seq_len = int(seq_len)
        self.fields = tuple(fields or dataset.fields)
        self.device = device
        self.generator = generator
        self.avoid_done = avoid_done
        self.sequence: Dict[str, torch.Tensor] = {}
        self.index = 0
        self.reset()

    def reset(self) -> Dict[str, torch.Tensor]:
        self.sequence = self.dataset.sample_sequence(
            self.seq_len,
            fields=self.fields,
            device=self.device,
            generator=self.generator,
            avoid_done=self.avoid_done,
        )
        self.index = 0
        return self.current()

    def current(self) -> Dict[str, torch.Tensor]:
        return {
            key: value[self.index]
            for key, value in self.sequence.items()
            if torch.is_tensor(value)
        }

    def step(self) -> Dict[str, torch.Tensor]:
        self.index += 1
        if self.index >= self.seq_len:
            self.reset()
        return self.current()

    @property
    def sequence_info(self) -> Mapping:
        return self.sequence.get("sequence_info", {})


def split_policy_srl_obs(
    srl_obs: torch.Tensor,
    spec: IsaacHRIObsSpec = IsaacHRIObsSpec(),
) -> Dict[str, torch.Tensor]:
    """Split 198D policy obs into stacked per-frame fields and task command.

    Frame layout after removing root_h and local_root_vel:
      0:3   local_root_ang_vel
      3:6   euler_err
      6:12  srl_dof_pos
      12:18 srl_dof_vel
      18:24 prev_action
      24:25 sin_phase
      25:26 cos_phase
      26:29 humanoid_euler_err
      29:35 load_cell_force
      35:39 human_leg_pitch
      -3:   task command [target_vel_x, target_ang_vel_z, target_height]
    """
    if srl_obs.shape[-1] != spec.policy_obs_dim:
        raise ValueError(f"Expected policy obs dim {spec.policy_obs_dim}, got {srl_obs.shape[-1]}")

    frame_part = srl_obs[..., : spec.frame_stack * spec.policy_frame_dim]
    frames = frame_part.reshape(*srl_obs.shape[:-1], spec.frame_stack, spec.policy_frame_dim)
    task_cmd = srl_obs[..., -spec.task_dim :]

    return {
        "frames": frames,
        "task_cmd": task_cmd,
        "local_root_ang_vel": frames[..., 0:3],
        "euler_err": frames[..., 3:6],
        "srl_dof_pos": frames[..., 6:12],
        "srl_dof_vel": frames[..., 12:18],
        "prev_action": frames[..., 18:24],
        "sin_phase": frames[..., 24:25],
        "cos_phase": frames[..., 25:26],
        "humanoid_euler_err": frames[..., 26:29],
        "load_cell_force": frames[..., 29:35],
        "human_leg_pitch": frames[..., 35:39],
    }


def split_full_srl_obs(
    srl_full_obs: torch.Tensor,
    spec: IsaacHRIObsSpec = IsaacHRIObsSpec(),
) -> Dict[str, torch.Tensor]:
    """Split 218D full obs into stacked frames and task command.

    The full frame is the original 43D IsaacGym SRL obs. Its first four fields
    are root_h and local_root_vel, which are not used by the deployable 198D
    policy obs.
    """
    if srl_full_obs.shape[-1] != spec.full_obs_dim:
        raise ValueError(f"Expected full obs dim {spec.full_obs_dim}, got {srl_full_obs.shape[-1]}")

    frame_part = srl_full_obs[..., : spec.frame_stack * spec.full_frame_dim]
    frames = frame_part.reshape(*srl_full_obs.shape[:-1], spec.frame_stack, spec.full_frame_dim)
    task_cmd = srl_full_obs[..., -spec.task_dim :]

    return {
        "frames": frames,
        "task_cmd": task_cmd,
        "root_h": frames[..., 0:1],
        "local_root_vel": frames[..., 1:4],
        "local_root_ang_vel": frames[..., 4:7],
        "euler_err": frames[..., 7:10],
        "srl_dof_pos": frames[..., 10:16],
        "srl_dof_vel": frames[..., 16:22],
        "prev_action": frames[..., 22:28],
        "sin_phase": frames[..., 28:29],
        "cos_phase": frames[..., 29:30],
        "humanoid_euler_err": frames[..., 30:33],
        "load_cell_force": frames[..., 33:39],
        "human_leg_pitch": frames[..., 39:43],
    }


def load_dataset(data_dir: str, **kwargs) -> IsaacHRIChunkDataset:
    return IsaacHRIChunkDataset(data_dir=data_dir, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect and sample an IsaacGym SRL-HRI dataset.")
    parser.add_argument("data_dir")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dataset = IsaacHRIChunkDataset(args.data_dir)
    print(dataset.summary())
    batch = dataset.sample_batch(args.batch_size, fields=("srl_obs", "raw_mu_srl", "virtual_load_cell", "done"))
    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype, "nan=", torch.isnan(value.float()).any().item())
