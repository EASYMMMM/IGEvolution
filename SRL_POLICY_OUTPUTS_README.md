# SRL 控制输出数据说明

## 1. 文档用途

本文档用于说明仿真导出的 SRL 控制网络输出文件如何使用，主要面向：

- 电机调试
- 板载控制回放
- 仿真与硬件输出对比

相关文件：

- 输出数据文件：[srl_policy_outputs_env0.csv](/home/ps/mly/IGEvolution/isaacgymenvs/run_data/srl_policy_outputs_env0.csv)
- SRL 模型结构文件：[srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml)

这份 CSV 是在 `test=True` 的演示/测试模式下生成的，当前只记录 `env0` 的 6 个 SRL 电机调试数据。

## 2. CSV 文件内容说明

CSV 中各列的含义如下：

- `episode`：第几个 episode，从 `1` 开始计数
- `step`：当前 episode 内的控制步编号，从 `0` 开始
- `time_sec`：当前控制步对应的仿真时间，单位秒
- `action_0 ~ action_5`：真正送入 SRL 控制链路的动作值
- `pd_target_0 ~ pd_target_5`：由动作映射得到的 SRL 目标关节角
- `torque_0 ~ torque_5`：PD 计算后的 SRL 关节力矩
- `joint_pos_0 ~ joint_pos_5`：仿真中该步结束后 SRL 六关节的实际关节位置

如果你的目标是：

- 做硬件实际回放：优先使用 `pd_target_*` 或 `torque_*`
- 看 SRL 控制链是否合理：一起对比 `action_*`、`pd_target_*`、`torque_*`、`joint_pos_*`

## 3. 序号与 SRL 关节的对应关系

CSV 中 6 个 SRL 输出维度，与 SRL 六个驱动关节一一对应，顺序如下：

| 序号 | CSV 后缀 | 关节名 | 左右侧 | 含义 |
|---|---|---|---|---|
| 0 | `_0` | `left_hip_x_joint` | 左侧 | 髋关节 X，自由度可理解为外展/内收 |
| 1 | `_1` | `left_hip_y_joint` | 左侧 | 髋关节 Y，自由度可理解为前后摆 |
| 2 | `_2` | `left_knee_joint` | 左侧 | 膝关节 |
| 3 | `_3` | `right_hip_x_joint` | 右侧 | 髋关节 X，自由度可理解为外展/内收 |
| 4 | `_4` | `right_hip_y_joint` | 右侧 | 髋关节 Y，自由度可理解为前后摆 |
| 5 | `_5` | `right_knee_joint` | 右侧 | 膝关节 |

这些关节定义见：

- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:253)
- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:271)
- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:291)
- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:323)
- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:341)
- [srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml:359)

## 4. 每个关节的机械运动范围

下面是 XML 中定义的 SRL 六个驱动关节机械范围。

| 序号 | 关节名 | 角度范围（deg） | 角度范围（rad） |
|---|---|---|---|
| 0 | `left_hip_x_joint` | `[-45.0, 45.0]` | `[-0.7854, 0.7854]` |
| 1 | `left_hip_y_joint` | `[-60.16, 60.16]` | `[-1.0500, 1.0500]` |
| 2 | `left_knee_joint` | `[-60.16, 60.16]` | `[-1.0500, 1.0500]` |
| 3 | `right_hip_x_joint` | `[-45.0, 45.0]` | `[-0.7854, 0.7854]` |
| 4 | `right_hip_y_joint` | `[-60.16, 60.16]` | `[-1.0500, 1.0500]` |
| 5 | `right_knee_joint` | `[-60.16, 60.16]` | `[-1.0500, 1.0500]` |

这些范围应当视为硬件侧的安全极限范围。

## 5. 时间间隔是多少

任务配置文件 [SRL_Real_HRI.yaml](/home/ps/mly/IGEvolution/isaacgymenvs/cfg/task/SRL_Real_HRI.yaml:1) 中定义了：

- `sim.dt = 0.005 s`
- `controlFrequencyInv = 3`

因此实际控制周期为：

`control_dt = 3 * 0.005 = 0.015 s`

也就是说：

- 每一行数据之间的时间间隔是 `15 ms`
- 控制频率约为 `66.67 Hz`

对于单个 episode，可以按下面方式恢复时间轴：

`time_sec = step * 0.015`

注意：

- 这个公式只在同一个 `episode` 内成立
- 如果跨 episode，需要同时结合 `episode` 和 `step` 来看

## 6. 四组数据分别处在控制链路的哪一层

这份 CSV 现在记录的是同一步的四层数据：

- `action_*`：神经网络输出并实际送入环境的动作值
- `pd_target_*`：动作经过映射后的目标关节角
- `torque_*`：PD 控制器根据目标角、当前角度和角速度计算出的力矩
- `joint_pos_*`：仿真积分之后，该步末端真实达到的关节位置

因此，这份 CSV 不再只是“神经网络输出”，而是完整覆盖了：

`动作 -> 目标角 -> 力矩 -> 实际关节位置`

其中：

- `action_*` 仍然是归一化控制量
- `pd_target_*` 和 `joint_pos_*` 的单位是 `rad`
- `torque_*` 的单位可按仿真力矩单位理解为 `N·m`

## 7. SRL 默认关节角

环境中 SRL 的默认关节角定义如下：

| 序号 | 关节名 | 默认角（rad） |
|---|---|---|
| 0 | `left_hip_x_joint` | `0.0` |
| 1 | `left_hip_y_joint` | `-0.1` |
| 2 | `left_knee_joint` | `0.35` |
| 3 | `right_hip_x_joint` | `0.0` |
| 4 | `right_hip_y_joint` | `-0.1` |
| 5 | `right_knee_joint` | `0.35` |

定义位置见：

- [srl_real_hri_base.py](/home/ps/mly/IGEvolution/isaacgymenvs/tasks/SRLEvo/srl_real_hri_base.py:121)

## 8. 动作如何转换成目标关节角

环境内部将策略动作转换为 PD 目标角，核心公式是：

`pd_target = offset + scale * action`

对应代码见：

- [srl_real_hri_base.py](/home/ps/mly/IGEvolution/isaacgymenvs/tasks/SRLEvo/srl_real_hri_base.py:1236)

对于当前这套 SRL 六关节，近似换算关系可以写成：

| 序号 | 关节名 | 动作到目标角的换算 |
|---|---|---|
| 0 | `left_hip_x_joint` | `q_ref = 0.0 + 0.7069 * action` |
| 1 | `left_hip_y_joint` | `q_ref = -0.1 + 0.9450 * action` |
| 2 | `left_knee_joint` | `q_ref = 0.35 + 0.9450 * action` |
| 3 | `right_hip_x_joint` | `q_ref = 0.0 + 0.7069 * action` |
| 4 | `right_hip_y_joint` | `q_ref = -0.1 + 0.9450 * action` |
| 5 | `right_knee_joint` | `q_ref = 0.35 + 0.9450 * action` |

然后，目标角还会再被夹紧到机械极限范围内。

## 9. 每个关节的目标角范围

结合上面的换算关系，可以得到动作 `action ∈ [-1, 1]` 时的目标角范围大致如下：

| 序号 | 关节名 | 目标角范围（未最终限位前） |
|---|---|---|
| 0 | `left_hip_x_joint` | `[-0.7069, 0.7069] rad` |
| 1 | `left_hip_y_joint` | `[-1.0450, 0.8450] rad` |
| 2 | `left_knee_joint` | `[-0.5950, 1.2950] rad` |
| 3 | `right_hip_x_joint` | `[-0.7069, 0.7069] rad` |
| 4 | `right_hip_y_joint` | `[-1.0450, 0.8450] rad` |
| 5 | `right_knee_joint` | `[-0.5950, 1.2950] rad` |

注意：

- `knee` 关节正方向较大时，会更容易碰到机械上限
- 最终真正的目标角还会被限制在 XML 机械范围内

## 10. 如何用于硬件电机调试

建议按照下面的方式使用这份 CSV。

### 10.1 推荐使用哪一列

如果你的目的是“复现策略最原始的输出”，看：

- `action_0 ~ action_5`

如果你的目的是“复现目标关节角参考”，看：

- `pd_target_0 ~ pd_target_5`

如果你的目的是“复现力矩控制输入”，看：

- `torque_0 ~ torque_5`

如果你的目的是“看实际机构在仿真里的响应结果”，看：

- `joint_pos_0 ~ joint_pos_5`

### 10.2 推荐的回放流程

建议回放步骤如下：

1. 从 CSV 中读取一行
2. 选取你要使用的数据层：
   `action_*`、`pd_target_*`、`torque_*`、`joint_pos_*`
3. 如果硬件控制器是位置环，优先用 `pd_target_*`
4. 如果硬件控制器是力矩环，优先用 `torque_*`
5. 在硬件侧再次做一层机械限位保护
6. 按 `15 ms` 的固定控制周期下发给电机控制器

### 10.3 为什么推荐再做一层硬件限位

虽然仿真里已经做过限位，但硬件侧仍然建议保留：

- 编码器角度限位
- 目标角限位
- 速度限位
- 电流/力矩限位

原因是：

- 仿真和硬件存在模型误差
- 零位标定可能有偏差
- 实际机构装配存在左右差异和柔性误差

## 11. 一个换算示例

如果某一行数据中：

`action_1 = 0.5`

那么对于 `left_hip_y_joint`：

`q_ref = -0.1 + 0.9450 * 0.5 = 0.3725 rad`

这个值落在机械允许范围内，因此该例中不需要额外裁剪。

## 12. 使用时需要注意的限制

这份 CSV 有以下限制：

- 只记录了 `env0`
- 只记录了 SRL 六个驱动关节，不包含 humanoid 关节
- 没有记录 SRL 网络输入
- 没有记录真实硬件编码器反馈
- 它是仿真播放过程中的控制链路日志，不是硬件实测日志

## 13. 建议补充给硬件同学的信息

如果你要把这份数据直接交给硬件或控制同学，建议同时说明以下几点：

- 控制周期固定为 `15 ms`
- 六个输出的顺序是“左腿三关节 + 右腿三关节”
- `action_*` 不是关节角，是动作值
- `pd_target_*` 才是目标角参考
- `joint_pos_*` 是仿真里该步结束后的真实位置
- 首次上电建议做缓启动和平滑滤波
- 尤其注意膝关节正方向容易接近饱和

## 14. 相关源码位置

- 输出数据文件：[srl_policy_outputs_env0.csv](/home/ps/mly/IGEvolution/isaacgymenvs/run_data/srl_policy_outputs_env0.csv)
- 模型结构文件：[srl_real_hri_v1_HXYK_175_mesh.xml](/home/ps/mly/IGEvolution/assets/mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml)
- 任务配置文件：[SRL_Real_HRI.yaml](/home/ps/mly/IGEvolution/isaacgymenvs/cfg/task/SRL_Real_HRI.yaml:1)
- 动作范围构建逻辑：[srl_real_hri_base.py](/home/ps/mly/IGEvolution/isaacgymenvs/tasks/SRLEvo/srl_real_hri_base.py:636)
- 动作到 PD 目标角转换逻辑：[srl_real_hri_base.py](/home/ps/mly/IGEvolution/isaacgymenvs/tasks/SRLEvo/srl_real_hri_base.py:1236)
