"""Minimal v2 env creator that resolves tasks from ``tasks_v2``."""

import os

from isaacgymenvs.tasks_v2 import isaacgym_task_map_v2


def get_rlgames_env_creator_v2(
        seed,
        task_config,
        task_name,
        sim_device,
        rl_device,
        graphics_device_id,
        headless,
        multi_gpu=False,
        post_create_hook=None,
        virtual_screen_capture=False,
        force_render=False,
):
    del seed  # kept for signature parity with the original helper

    def create_rlgpu_env():
        if multi_gpu:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

            _sim_device = f"cuda:{local_rank}"
            _rl_device = f"cuda:{local_rank}"
            task_config["rank"] = local_rank
            task_config["rl_device"] = _rl_device
        else:
            _sim_device = sim_device
            _rl_device = rl_device

        env = isaacgym_task_map_v2[task_name](
            cfg=task_config,
            rl_device=_rl_device,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env

    return create_rlgpu_env
