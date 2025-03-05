'''
检查线性关节
'''
 

import math
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch
import matplotlib.pyplot as plt

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# simple asset descriptor for selecting from a list


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("mjcf/humanoid_srl/hsrl_mode1_prismatic_test.xml", False),
    AssetDesc("mjcf/humanoid_srl/hsrl_mode1.xml", False),
    # AssetDesc("mjcf/amp_humanoid_srl_V2_1.xml", False),
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
]


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])



# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "assets"
asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
# 固定基座位置
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)
dof_props["driveMode"] = gymapi.DOF_MODE_POS



# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default
    dof_positions[i] = defaults[i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 8
num_per_row = 4
spacing = 4
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


# position the camera
cam_pos = gymapi.Vec3(10, 10, 5)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

slide_x_idx = gym.find_asset_dof_index(asset, 'sensor_x' )
slide_y_idx = gym.find_asset_dof_index(asset, 'sensor_y' )
slide_z_idx = gym.find_asset_dof_index(asset, 'sensor_z' )


srl_board_idx = gym.find_asset_rigid_body_index(asset, 'SRL' )
srl_root_idx = gym.find_asset_rigid_body_index(asset, 'SRL_root' )
right_foot_idx = gym.find_asset_rigid_body_index(asset, 'SRL_right_end' )
left_foot_idx = gym.find_asset_rigid_body_index(asset, 'SRL_left_end' )
sensor_pose = gymapi.Transform()

# sensor props
sensor_props = gymapi.ForceSensorProperties()
sensor_props.enable_forward_dynamics_forces = True
sensor_props.enable_constraint_solver_forces = True
sensor_props.use_world_frame = False

right_foot_ssidx = gym.create_asset_force_sensor(asset, right_foot_idx, sensor_pose, sensor_props)
left_foot_ssidx  = gym.create_asset_force_sensor(asset, left_foot_idx, sensor_pose, sensor_props)
srl_board_ssidx  = gym.create_asset_force_sensor(asset, srl_board_idx, sensor_pose, sensor_props)
srl_root_ssidx   = gym.create_asset_force_sensor(asset, srl_root_idx, sensor_pose, sensor_props)



# cache useful handles
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0,  0.0, 1.32)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
    gym.enable_actor_dof_force_sensors(env, actor_handle)

    dof_prop = gym.get_asset_dof_properties(asset)

    # 被动控制
    # dof_prop["driveMode"][slide_x_idx] = gymapi.DOF_MODE_NONE
    # dof_prop["driveMode"][slide_y_idx] = gymapi.DOF_MODE_NONE
    # dof_prop["driveMode"][slide_z_idx] = gymapi.DOF_MODE_NONE
    # dof_prop["stiffness"][slide_x_idx] = 200.0
    # dof_prop["stiffness"][slide_y_idx] = 200.0
    # dof_prop["stiffness"][slide_z_idx] = 200.0
    # dof_prop["damping"][slide_x_idx] = 50.0
    # dof_prop["damping"][slide_y_idx] = 50.0
    # dof_prop["damping"][slide_z_idx] = 50.0
    gym.set_actor_dof_properties(env, actor_handle, dof_prop)

    # 位控
    # dof_prop["driveMode"] = gymapi.DOF_MODE_POS
    # gym.set_actor_dof_properties(env, actor_handle, dof_prop)

    # 力控
    dof_prop["driveMode"][slide_x_idx] = gymapi.DOF_MODE_EFFORT
    dof_prop["driveMode"][slide_y_idx] = gymapi.DOF_MODE_EFFORT
    dof_prop["driveMode"][slide_z_idx] = gymapi.DOF_MODE_EFFORT
    dof_prop["stiffness"][slide_x_idx] = 0
    dof_prop["stiffness"][slide_y_idx] = 0
    dof_prop["stiffness"][slide_z_idx] = 0
    dof_prop["damping"][slide_x_idx] = 0
    dof_prop["damping"][slide_y_idx] = 0
    dof_prop["damping"][slide_z_idx] = 0
    gym.set_actor_dof_properties(env, actor_handle, dof_prop)
    

# rigid body properties
rigid_body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
rigid_body_names = gym.get_actor_rigid_body_names(env, actor_handle)
num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
for i in range(num_bodies):
    print("Rigid Body %d" % i)
    print("  Name:     '%s'" % rigid_body_names[i])
    print("  Mass:  %r" % rigid_body_props[i].mass)


# for i in range(num_envs):
#     gym.set_actor_dof_properties(envs[i], actor_handles[i], dof_props)

gym.prepare_sim(sim)
 
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
_dof_state = gymtorch.wrap_tensor(dof_state_tensor)
_dof_pos = _dof_state.view(num_envs, num_dofs, 2)[..., 0]
_dof_vel = _dof_state.view(num_envs, num_dofs, 2)[..., 1]

_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)
gym.refresh_actor_root_state_tensor(sim)
root_positions = root_tensor[:,0:3]
root_orientations = root_tensor[:,3:7]
root_linvels = root_tensor[:,7:10]
root_angvels = root_tensor[:, 10:13]

_dof_force_tensor = gym.acquire_dof_force_tensor(sim)
_rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
_sensor_tensor = gym.acquire_force_sensor_tensor(sim)
dof_force_tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(num_envs, num_dofs)
rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state)
sensors_per_env = 4
vec_sensor_tensor = gymtorch.wrap_tensor(_sensor_tensor).view(num_envs, sensors_per_env, 6)


rigid_body_pos =  rigid_body_state.view(num_envs, num_bodies, 13)[..., 0:3]
gym.refresh_dof_state_tensor(sim)
gym.refresh_dof_force_tensor(sim)

pd_tar = torch.zeros(num_envs, num_dofs)
step_count = 0
force_data = []
sensor_data = []
sensor_data_x = []
sensor_data_y = []
position_data = []





while not gym.query_viewer_has_closed(viewer):
    # get_actor_dof_forces
    force_value = dof_force_tensor[0, slide_z_idx].item()       # 关节受力
    sensor_value_x = vec_sensor_tensor[0,srl_board_ssidx, 0].item() 
    sensor_value_y = vec_sensor_tensor[0,srl_board_ssidx, 1].item() 
    sensor_value = vec_sensor_tensor[0,srl_board_ssidx, 2].item() 
    position_value = rigid_body_pos[0, srl_root_idx, 2].item()   # 第二个刚体的y轴位移
    print(f"Step {step_count} | Joint: {force_value}, Sensor: {sensor_value}, Pos: {position_value}")
    force_data.append(force_value)
    sensor_data_x.append(sensor_value_x)
    sensor_data_y.append(sensor_value_y)
    sensor_data.append(sensor_value)
    position_data.append(position_value)
    step_count += 1

    # 添加外部力
    if step_count > 240:
        forces = torch.zeros((num_envs, num_bodies, 3),   dtype=torch.float)
        torques = torch.zeros((num_envs, num_bodies, 3),  dtype=torch.float)
        forces[:, right_foot_idx, 2] = 200
        forces[:, left_foot_idx, 2] = 200
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)


    pd_tar[:,slide_z_idx] = 0
    pd_tar[:,2] = 0
    pd_tar[:,3] = 0
    # 将 pd_tar 转换为 Tensor
    pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)

    # 位控
    # result = gym.set_dof_position_target_tensor( sim, pd_tar_tensor)

    # 力控
    for i in range(num_envs):
        efforts = np.full(num_dofs, 0).astype(np.float32)
        gym.apply_actor_dof_efforts(envs[i], actor_handles[i], efforts)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

 
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    gym.refresh_force_sensor_tensor(sim)
    gym.refresh_dof_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)



print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# 时间轴 (dt * step_count)
time_axis = np.arange(0, len(force_data) * dt, dt)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左边 Y 轴：关节受力
ax1.plot(time_axis, force_data, label="Joint Force (N)", color="red", linewidth=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Joint Force (N)", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid(True)
ax1.set_ylim(-15, 5)


# 右边 Y 轴：刚体位移
ax2 = ax1.twinx()  # 创建共享 X 轴的第二个 Y 轴
ax2.plot(time_axis, position_data, label="Body Position (m)", color="blue", linewidth=2)
ax2.set_ylabel("Body Position (m)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.set_ylim(1.3, 1.4)

# 标题和图例
plt.title("Joint Force and Body Position over Time")
fig.tight_layout()

 

fig2, ax3 = plt.subplots(figsize=(10, 6))
# 左边 Y 轴：关节受力
ax3.plot(time_axis, force_data,  label="Joint Force (N)",  linewidth=2)
ax3.plot(time_axis, sensor_data, label="Sensor Force (N)",  linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Force (N)", color="red")
ax3.legend()
ax3.tick_params(axis="y", labelcolor="red")
ax3.grid(True)

# 标题和图例
plt.title("Sensor Force vs Joint Force")
fig.tight_layout()

plt.show()


fig3, ax4 = plt.subplots(figsize=(10, 6))
# 左边 Y 轴：关节受力
ax4.plot(time_axis, sensor_data_x, label="Sensor Force X (N)",  linewidth=2)
ax4.plot(time_axis, sensor_data_y, label="Sensor Force Y (N)",  linewidth=2)
ax4.plot(time_axis, sensor_data,   label="Sensor Force Z (N)",  linewidth=2)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Force (N)", color="red")
ax4.legend()
ax4.tick_params(axis="y", labelcolor="red")
ax4.grid(True)

# 标题和图例
plt.title("Rigid Body Sensor Force ")
fig.tight_layout()

plt.show()