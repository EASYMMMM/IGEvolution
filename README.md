# IG Evolution

## 0. 常用命令
- **添加虚拟环境路径：**

   `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`

   需要`export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/XXX/lib`  
   其中`XXX`为虚拟环境名称。
   `export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/rlgpu/lib`   
   `export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/Mrlgpu/lib`
   `export LD_LIBRARY_PATH=/home/user/miniconda3/envs/mly_isaacgym/lib`

- **登录wandb：**
   `wandb login` 
   随后输出自己账户的API。
   仅在当前终端登录账号：
   `export WANDB_API_KEY=d9e147c0c0f29ad02ca38e65742ce8ce2bbd52ab`   

- **重启todesk：**
   `sudo systemctl stop  todeskd.service`
   `sudo systemctl start todeskd.service`

- **通过SCP从服务器传输训练模型：**
   `scp -r  user@172.18.41.167:/home/user/mly/IGEvolution/isaacgymenvs/runs/SRL_walk_v1.5.15.0_25-11-01-51  /home/ps/pan1/files/mly/IsaacGymEvo/isaacgymenvs/runs`
   
## 1. 环境配置
[Isaac Gym环境安装和四足机器人模型的训练-CSDN博客](https://blog.csdn.net/weixin_44061195/article/details/131830133?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-2-131830133-blog-124605383.235^v38^pc_relevant_sort&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-2-131830133-blog-124605383.235^v38^pc_relevant_sort&utm_relevant_index=5)

### 1.1 配置isaacgym虚拟环境
在官网下载最新的文件包[Isaac Gym - Preview Release](https://developer.nvidia.com/isaac-gym)，注意需要登陆。   
在`create_conda_env_rlgpu.sh`文件中更改`XXX`为虚拟环境名：   
   ```sh
   # should match env name from YAML 
      ENV_NAME=mly_isaacgym
   ```
   随后在`python/rlgpu_conda_env.yaml`文件中更改为同样的虚拟环境名：
   ```yaml
   name: mly_isaacgym
   ```
   随后在`isaacgym`根目录中创建虚拟环境。
   ```bash
   # 在文件根目录里运行
   ./create_conda_env_rlgpu.sh
   # 激活环境
   conda activate rlgpu
   # 正常情况下这时候可以运行实例程序/isaacgym/python/examples/joint_monkey.py
   python joint_monkey.py
   # 可以通过--asset_id命令控制显示的模型
   python joint_monkey.py --asset_id=6
   ```
   到这里IsaacGym就安装好了.

   `./create_conda_env_rlgpu.sh`要等很久是正常的.

### 1.2 报错处理

   出现这样的报错：

   `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`

   需要`sudo apt install libpython3.7`

   或者`export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/rlgpu/lib`
   `export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/Mrlgpu/lib`

   尝试一下就行 大概率后者设置路径有效



## 2. IsaacGym基础训练环境安装

在本仓库内先激活虚拟环境，随后：

```bash
pip install -e .
```



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












