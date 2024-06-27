# ISAAC GYM

## 1. 环境配置

配置整体来说是比较简单的，遇到的几个问题都能看教程解决

[Isaac Gym环境安装和四足机器人模型的训练-CSDN博客](https://blog.csdn.net/weixin_44061195/article/details/131830133?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-2-131830133-blog-124605383.235^v38^pc_relevant_sort&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-2-131830133-blog-124605383.235^v38^pc_relevant_sort&utm_relevant_index=5)

1.在官网下载最新的文件包[Isaac Gym - Preview Release](https://developer.nvidia.com/isaac-gym)，注意需要登陆。

其中assets是模型材料位置
其中docs是说明网站位置
其中python是演示程序位置
可以根据说明文档安装方式安装
这里只介绍其中conda安装的方法

```
# 在文件根目录里运行
./create_conda_env_rlgpu.sh
# 激活环境
conda activate rlgpu
# 正常情况下这时候可以运行实例程序/isaacgym/python/examples/joint_monkey.py
python joint_monkey.py
# 可以通过--asset_id命令控制显示的模型
python joint_monkey.py --asset_id=6

```

到这里IsaacGym就安装好了

### 1.1 根据教程配置Isaac Gym

`./create_conda_env_rlgpu.sh`要等很久是正常的

### 1.2 无论何时遇到 这个很关键

出现这样的报错：

`
ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`

需要`sudo apt install libpython3.7`

或者`export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/rlgpu/lib`
`export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/Mrlgpu/lib`

登录wandb：
`wandb login` 
随后输出自己账户的API。
仅在当前终端登录账号：
`export WANDB_API_KEY=d9e147c0c0f29ad02ca38e65742ce8ce2bbd52ab`
尝试一下就行 大概率后者设置路径有效

如果报以下错误是因为模型文件URDF文件中mesh文件的地址出错，找不到模型文件导致的。建议可以直接写绝对地址。

```
[Error] [carb.gym.plugin] Failed to resolve visual mesh '/isaacgym/Quadruped/legged_gym-master/resources/robots/meshes/anymal/trunk.stl'

```

## 2. IsaacGym基础训练环境安装

GitHub上下载好 进文件内

```
conda activate rlgpu
pip install -e .
```

有几个装不上的多试几次，重启终端、电脑

这里装完之后pytorch好像得重装

Isaac gym envs也是根据教程 但是训练模型时候遇到了问题，训练的gui一闪而过之后`run time error nvrtc: invalid value for --gpu-architecture(-arch)`

尝试重装pytorch可以 然后重新设置一下上面的那个路径

pytorch 1.12 cuda11.6

到train.py 的文件夹路径内训练

```
python train.py task=Ant
# 不显示动画只训练
python train.py task=Ant headless=True
# 测试训练模型的效果，num_envs是同时进行训练的模型数量
python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64

```

## 3. IsaacGym 基础训练

- **训练基础例程**

  ```bash
  python train.py task=Cartpole
  ```

  ```bash
  python train.py task=Ant headless=True
  ```

  `headless`表示关闭训练时的渲染。

- **从checkpoint继续训练**

  Checkpoints保存在文件夹 `runs/EXPERIMENT_NAME/nn` 下，其中`EXPERIMENT_NAME` 
  默认为task名。

  要加载一个checkpoint然后继续训练，可以使用`checkpoint`参数：

  ```bash
  python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth
  ```

- **检查训练结果**

  若要仅加载训练好的模型（checkpoint）并查看训练结果，使用`test=True`参数。可以同时限制环境数，以避免过度渲染：

  ```bash
  python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64
  ```

  Note that If there are special characters such as `[` or `=` in the checkpoint names, 
  you will need to escape them and put quotes around the string. For example,
  `checkpoint="./runs/Ant/nn/last_Antep\=501rew\[5981.31\].pth"`

## 3. AMP训练Humanoid

- **存在BUG**: `home/ps/IssacGymEnvs-main/issacgymenvs/learning/commen_agent.py`第98行：`self.seq_len`

本示例训练一个模拟人模型，以模仿存储在 mocap 数据中的不同预录人体动画 - 走路、跑步和后空翻。

可以使用命令行参数 "task=HumanoidAMP "启动它。可以在任务配置中使用 `motion_file` 设置要训练的动画文件。注意：在测试模式下，查看器摄像头会从第一个环境开始跟随Humanoid。这可以在环境 yaml 配置中通过设置 `cameraFollow=False` 进行更改，或在命令行中通过 hydra 覆盖进行更改，如下所示：`++task.env.cameraFollow=False`。



本软件源包含 CMU 动作捕捉库 (http://mocap.cs.cmu.edu/) 中的一些动作，但可以使用 poselib `fbx_importer.py`将其他动画从 FBX 转换为可训练的格式。您可以在 `isaacgymenvs/tasks/amp/poselib/README.md` 中了解有关 poselib 和该转换工具的更多信息。



```bash
python train.py task=HumanoidAMP ++task.env.motion_file=amp_humanoid_run.npy experiment=AMP_run rl_device=cuda:1 sim_device=cuda:1
```





# 外肢体形态自生成

## 1. 研究目标

​    使用人工智能方法， 在仿真环境中实现SRL的形态自生成及优化。给定一个任务（行走、跑步、坐站），迭代优化得到外肢体及其对应控制器。

## 2. 技术难点及方法

- 保证人体模型在给定任务下的运动为真实人类运动

  - AMP方法

    基于真实的运动捕捉数据。Humanoid智能体的奖励函数中，通过GAIL添加一项**真实动作模拟项**，能够在执行目标任务时保持对给定数据集的模仿。由于引入动作约束的方式是奖励函数，其学习到的动作并非固定。当有其他目标约束的奖励项时，智能体会按照强化学习的逻辑进行学习。譬如实现以给定动作的风格完成另一种任务。

- 构建人体模型和SRL的多智能体强化学习模型

  - 多智能体强化学习

- 对大范围设计空间的探索

  ```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
  
  ```

•研究目标：SRL的形态设计

•研究方法和内容：

•人控制器

•SRL控制器

•形态自生成

•形态优化器

•联合优化器

•任务设计

•约束设计

•难点

•动态交互时人类模型保持类人运动

•创新点 研究重点：

•1. SRL（人机协作、辅助机器人）的AI自动设计框架

•2. 对现有的AI设计机器人方法进一步改进，与人形结合，进一步考虑实际意义

•3. 离散+连续庞大空间的高效搜索




(base) ps@ps:~$ ip addr show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: enp4s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    link/ether 74:56:3c:77:09:19 brd ff:ff:ff:ff:ff:ff
    inet 10.5.28.91/22 brd 10.5.31.255 scope global dynamic noprefixroute enp4s0
       valid_lft 481480sec preferred_lft 481480sec
    inet6 2400:dd01:1032:8:18a8:eb80:8210:e8a9/64 scope global temporary dynamic 
       valid_lft 567821sec preferred_lft 49362sec
    inet6 2400:dd01:1032:8::205/128 scope global dynamic noprefixroute 
       valid_lft 225622sec preferred_lft 52822sec
    inet6 2400:dd01:1032:8:9510:fca9:21ae:85e0/64 scope global temporary deprecated dynamic 
       valid_lft 481482sec preferred_lft 0sec
    inet6 2400:dd01:1032:8:50fc:bb27:e6e9:caac/64 scope global dynamic mngtmpaddr noprefixroute 
       valid_lft 2591673sec preferred_lft 604473sec
    inet6 fe80::c8b5:fba8:61a5:f1ca/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
3: tailscale0: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1280 qdisc fq_codel state UNKNOWN group default qlen 500
    link/none 
    inet 100.73.14.81/32 scope global tailscale0
       valid_lft forever preferred_lft forever
    inet6 fd7a:115c:a1e0::2ec9:e51/128 scope global 
       valid_lft forever preferred_lft forever
    inet6 fe80::ea2b:f783:d778:8c1c/64 scope link stable-privacy 
       valid_lft forever preferred_lft forever
4: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:e8:ea:a3:88 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft foreverv