# 可选 HMI 长历史编码器

默认配置为 `train.params.config.hmi_encoder.enabled=False`。开启后，编码器从五帧 SRL 观测中提取历史特征。固定形态策略使用 198 维输入，启用 `task.env.design_param_obs` 的策略在三个命令前额外读入七个形态参数，两种策略都输出六个动作。

## 文件职责与修改入口

| 文件 | 职责 |
| --- | --- |
| `config.py` | 全部开关、网络尺寸、窗口、预测时距、损失、学习率和渐进接入参数 |
| `layout.py` | 唯一依赖现有观测索引和仿真状态的适配器；更换观测排序时修改这里并升级 layout 版本 |
| `model.py` | 无仿真依赖的因果 TCN、15 维瓶颈、预测头、输入标准化 |
| `runtime.py` | 历史/reset、rollout 窗口重建、监督标签、校准、接入进度与编码器 checkpoint |
| `training.py` | 可选的 PPO 微批次梯度累积及辅助监督；原 PPO 更新函数保留 |
| `checkpoint.py` | 联合网络权重迁移和恢复，包括编码器与优化器 |

现有代码只保留几类接入点：训练器创建/采样/更新/保存加载，网络在归一化后注入 `actor_latent`，player 创建/重置/加载/推理。新功能可关闭；删除实现时还需移除这些显式接入点。已训练成依赖编码器的模型不能通过简单关闭开关来保证同等效果，应使用旧 stage4 checkpoint 或重新训练。

## 数据契约

- Actor 接收五帧各 39 维的观测，可选七个形态值，以及末尾的三个任务命令，共 198 或 205 维。HMI 模式在归一化后将这三个命令置零。
- 零力矩槽：`32:35,71:74,110:113,149:152,188:191`。这些槽承载同一个当前 15 维隐变量的五个分组，不再具有力矩/历史时刻语义。
- Encoder：64 帧、每帧 41 维，时间由旧到新；控制周期 .015 秒，首尾覆盖 .945 秒。
- 特征顺序：六个姿态角的 sin(6)、cos(6)、交互力(3)、人体四腿段 pitch(4)、SRL 角速度(3)、关节角/速度/过去动作(18)、有效标记(1)。角度来源及字段索引仅定义于 `layout.py`。
- 关节角/速度和力使用现有观测尺度，再经过编码器独立标准化。reset 填充帧归一化后仍为零。
- TCN：41→32，六层 kernel=2 的因果残差卷积，dilation=1/2/4/8/16/32，逐时间点通道 LayerNorm，32→32→15 Tanh；预测头 15→64→9。
- 两点卷积通过 `W[0] x[t-d] + W[1] x[t] + b` 的两个 `F.linear` 执行，不依赖 cuDNN。仍由原 `conv.weight`/`conv.bias` 持有参数，参数名称、形状、顺序和 checkpoint 版本不变；旧 TCN 权重及 Adam 状态可直接加载。保留左侧补零、原 LayerNorm 和残差。浮点计算顺序不同，不承诺逐位一致。
- 监督目标：当前、未来8步(.12秒)、16步(.24秒)人体 vx、vy、航向角速度。vx/vy 均以锚点时刻 SRL 航向坐标系表达；航向角速度由连续人体 yaw 的包角差分得到。
- 人体真实状态只用于 detached 监督标签，不进入编码器/Actor。reset 后第一帧 yaw 差分无效；跨 episode 或超出 rollout 的未来标签被屏蔽。
- 启用时要求 `srl_hmi_control=True`、`HMI_obs_enable=True`、原39维布局、五帧、集中式 critic、无 DAgger teacher、单 GPU、非循环 PPO。时序模型自身不使用跨批次循环隐状态。

## 更新与归一化

`alpha=0` 时，完整保留 stage4 归一化后的输入，包括可能非零的占位归一化值。逐渐替换为 `(1-alpha)*old_slots+alpha*z`，最终 alpha=1。旧 RMS 固定，critic/奖励继续读取原始观测，编码不会污染它们。

默认：前100个 rollout alpha=0；接下来300个 rollout 逐渐接入。前10个 rollout 在整轮 PPO 更新结束后校准编码器统计量，之后固定。alpha 在单轮 rollout/PPO 内固定。alpha=0 时 humanoid/SRL 仍更新，编码器通过辅助监督学习。

监督损失为每个时距单独按有效样本平均的 Huber：`L0 + 0.5*L8 + 0.25*L16`，附加系数默认1。PPO 梯度和辅助梯度都更新编码器。预测头只接受辅助监督。

微批次默认2048，累积成现有完整 PPO minibatch 后只执行一次 optimizer.step；各预测时距的有效样本分母使用完整 minibatch，避免 reset 和分块改变损失权重。历史只存前63帧加本轮T帧，按样本环境/时间索引重建，不能仅缓存 detached z。

镜像：重新编码整段镜像原始历史；不对 z 使用旧力矩符号规则。镜像标签 vx 保持、vy/yaw_rate 取反。保持现有动作对称损失。

默认 SRL lr=1e-4，Encoder lr=3e-4；humanoid 实际 lr=公共lr×0.3，即3e-5，并沿用 `train_humanoid_freq=10`。学习率调度更新时仍保持这个比例。

## Checkpoint 与调用方式

1. **从 stage4 开始**：`hsrl_checkpoint=<旧stage4.pth>`、`hmi_encoder.enabled=True`，`humanoid_checkpoint=False`、`srl_teacher_checkpoint=False`。完整载入旧联合网络、AMP统计量、集中式critic；新建编码器与优化器，alpha=0，开始新实验。不要把这个初始化误当作旧训练进度的续跑。
2. **初始化一个新形态的策略**。使用 `train.params.config.hsrl_checkpoint=<联合策略.pth>`。若已有编码器，设置 `train.params.config.hmi_encoder.enabled=True`，载入其权重、归一化统计和 alpha 接入进度。新候选从第一个训练 epoch 开始，并使用新的优化器。从 198 维输入迁移到 205 维时，Actor 和两种 critic 的新输入权重初始化为零，七个新输入的归一化均值为零、方差为一。
3. **恢复中断训练**。使用 `checkpoint=<带encoder.pth>` 和 `train.params.config.hsrl_checkpoint=False`，并开启编码器，使用原训练的观测布局。恢复策略、critic、优化器、归一化统计、接入进度、epoch 和 frame。`max_iterations` 是累计迭代上限，须大于已保存 epoch。
4. **推理**：`test=True checkpoint=<带encoder.pth> train.params.config.hmi_encoder.enabled=True`，同样开启两个环境HMI开关。player 从 checkpoint 内的配置构建编码器，载入权重/统计/alpha。使用已保存 alpha，绝不擅自设为1。
5. **模式检查**：加载含编码器的 checkpoint 却关闭功能会报错；启用编码器进行推理却加载旧 stage4 也会报错，避免悄悄使用随机编码器。

Encoder 作为顶层 `hmi_encoder` 字段保存，不增加旧 `model_srl` 的 state_dict 键。布局、时间步、架构、窗口和预测语义有版本/兼容性检查。恢复时允许调整微批次、学习率、humanoid学习率比例和辅助损失参数；结构/归一化/接入计划变化需要显式迁移。新 checkpoint 会记录调整后的配置。

恢复时重新开始仿真 episode 并清空历史；不承诺重现断点处的物理状态、随机数流或 AMP replay buffer。保存点处的策略、优化器和接入进度可以恢复。没有训练完成前，预测质量和控制改善程度仍未知。

## 验证

`test_srl_hmi_tcn.py` 对照原 Conv1d 实现，验证所有膨胀率、短序列补零、完整编码器输出/输入梯度/参数梯度，以及原模型和 Adam checkpoint 加载后的下一步更新。

GPU 性能验证（不创建仿真、不启动训练、不执行 optimizer.step）：

```bash
python tests/benchmark_hmi_tcn.py --output /tmp/hmi_tcn_benchmark.json
```

该脚本在同一 GPU、相同权重及 FP32/TF32关闭设置下比较原实现和矩阵实现；先验证 GPU 输出与梯度，再对每种形状预热，重复3次取中位数。分别测试8192样本推理和2048样本前向/反向，并记录临时显存峰值。编码器倍速不等于整轮 PPO/仿真倍速；矩阵实现也可能增加临时显存占用。

从仓库任意外部目录可运行独立启动脚本：

```bash
bash /home/joemeng/SRL/IGEvolution/run_srl_hmi_stage5.sh
```

它启动第五阶段，参数与 `srl_real_hri.bash` 第五阶段一致，并使用 `python -u` 及时输出日志。默认从原 stage4 初始化。要从第五阶段的中断点续训，请在命令末尾追加 `checkpoint=/绝对路径/第五阶段.pth train.params.config.hsrl_checkpoint=False`。

使用 srlgym 环境执行：

```bash
python -m unittest discover -s tests -p 'test_srl_hmi*.py' -v
```

测试不创建仿真或启动训练。使用真实 stage4 权重验证联合加载和 alpha=0 动作一致性；使用 CPU 合成数据验证因果性、reset、窗口/标签、镜像、PPO微批次梯度、冻结旧RMS、保存/恢复、player输出及模式拒绝。可用环境变量 `SRL_STAGE4_CHECKPOINT` 指定其他 stage4 文件。该文件不存在时，仅真实 checkpoint 测试会被跳过。
