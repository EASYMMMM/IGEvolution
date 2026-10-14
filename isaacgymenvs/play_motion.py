"""Isaac Gym AMP reference playback: --model smpl, capsule, or both.

This replays MotionLib poses; it does not run the learned muscle policy.
Requires the same Isaac Gym/PyTorch environment as SRL_Evo_train.py.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--motion", type=Path, default=project / "assets/amp/motions/P002_WALK_T01_walk_amp.npy")
    parser.add_argument("--model", choices=["smpl", "capsule", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graphics-device", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--fixed-camera", action="store_true")
    args = parser.parse_args()
    if not args.motion.is_file():
        parser.error(f"Motion file not found: {args.motion}")
    if args.speed <= 0:
        parser.error("Playback speed must be positive")
    # Isaac Gym must be imported before Torch.
    import isaacgym
    from isaacgym import gymapi, gymtorch
    import numpy as np
    import torch
    sys.path.insert(0, str(project))
    from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

    device = torch.device(args.device)
    gpu = device.type == "cuda"
    compute_id = (device.index or 0) if gpu else 0
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = gpu
    params.use_gpu_pipeline = gpu
    sim = gym.create_sim(compute_id, args.graphics_device, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym could not create the playback simulation")
    viewer = None
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
        models = ["capsule", "smpl"] if args.model == "both" else [args.model]
        actors, offsets = [], []
        for index, model in enumerate(models):
            options = gymapi.AssetOptions()
            options.disable_gravity = True
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            file = "mjcf/amp_humanoid_175_muscle" + ("_smpl" if model == "smpl" else "") + ".xml"
            asset = gym.load_asset(sim, str(project / "assets"), file, options)
            if asset is None or gym.get_asset_dof_count(asset) != 28:
                raise RuntimeError(f"Expected a 28-DOF humanoid: {file}")
            offset = (index - (len(models) - 1) / 2) * 1.3
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, offset, 1)
            actor = gym.create_actor(env, asset, pose, model, index, -1, 0)
            props = gym.get_actor_dof_properties(env, actor)
            props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            props["stiffness"].fill(0)
            props["damping"].fill(0)
            gym.set_actor_dof_properties(env, actor, props)
            actors.append(actor)
            offsets.append(offset)
            # Do not recolor SMPL: recoloring can expose the hidden collision geoms.
            if model == "capsule":
                for body in range(gym.get_asset_rigid_body_count(asset)):
                    gym.set_rigid_body_color(env, actor, body, gymapi.MESH_VISUAL, gymapi.Vec3(.47, .55, .69))
        key_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        keys = np.array([gym.find_actor_rigid_body_handle(env, actors[0], name) for name in key_names], dtype=np.int32)
        motion_lib = MotionLib(str(args.motion.resolve()), 28, keys, str(device))
        # Verify that the asset body order matches the skeleton used by AMP.
        tree_names = list(motion_lib.get_motion(0).skeleton_tree.node_names)
        actor_names = list(gym.get_actor_rigid_body_names(env, actors[0]))
        if actor_names[:len(tree_names)] != tree_names:
            raise RuntimeError("MJCF body order does not match the AMP motion skeleton")
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Isaac Gym could not create the viewer")
        gym.prepare_sim(sim)
        root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(len(actors), 28, 2)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        ids = np.zeros(len(actors), dtype=np.int32)
        duration = float(motion_lib.get_motion_length(np.array([0]))[0])
        time = 0.
        print(f"Playing {args.motion.name}: {duration:.3f}s, models={models}, speed={args.speed}")
        while not gym.query_viewer_has_closed(viewer):
            times = np.full(len(actors), time, dtype=np.float32)
            pos, quat, dof, _, _, _, _ = motion_lib.get_motion_state(ids, times)
            root_state[:, :3] = pos
            root_state[:, 1] += torch.tensor(offsets, dtype=torch.float32, device=device)
            root_state[:, 3:7] = quat
            root_state[:, 7:] = 0  # Keep imposed poses from drifting during propagation.
            dof_state[:, :, 0] = dof
            dof_state[:, :, 1] = 0
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            if not args.fixed_camera or time == 0:
                x, y = pos[0, :2].cpu().tolist()
                gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(x + 2.8, y - 4.2, 1.7),
                                         gymapi.Vec3(x, y, .9))
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            time = (time + params.dt * args.speed) % duration
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
