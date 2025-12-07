# IG Evolution

## 0. 常用命令
- **添加虚拟环境路径：**

   每次运行前需要`export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/XXX/lib`  
   其中`XXX`拟环境名称。  
   `export LD_LIBRARY_PATH=/home/ps/miniconda3/envs/Mrlgpu/bin`
   `export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib`     
   `export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/rlgpu/lib`      
   `export LD_LIBRARY_PATH=/home/ps/anaconda3/envs/Mrlgpu/lib`  
   `export LD_LIBRARY_PATH=/home/user/miniconda3/envs/igm/lib`  
   `export LD_LIBRARY_PATH=/home/zdh/anaconda3/envs/rlgpu/lib`  
   `export LD_LIBRARY_PATH=/home/pc/anaconda3/envs/Mrlgpu/lib`  

- **登录wandb：**  
   `wandb login` 
   随后输出自己账户的API。   
   仅在当前终端登录账号：  
   `export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace`   

- **解决Clash的网络问题：**    
  开启Clash后，可能会发生无法连接wandb：   
   `export http_proxy=http://127.0.0.1:7890`  
   `export https_proxy=http://127.0.0.1:7890`  

- **重启todesk：**  
   `sudo systemctl stop  todeskd.service`  
   `sudo systemctl start todeskd.service`  

- **通过SCP从服务器传输训练模型回本地：**  
   `scp -r  QH-MAIS:/nfs/IGEvolution/IGEvolution/isaacgymenvs/runs/TRO_SRL_v2.5.3.3_14-15-16-39  /home/zdh232/mly/IGEvolution/isaacgymenvs/runs`  
   `scp -r user@172.18.41.167:/home/user/mly/IGEvolution/isaacgymenvs/runs/SRL_walk_v2.0_A100_16-18-37-52  /home/zdh/mly/IGEvolution/isaacgymenvs/runs`

- **通过SCP从本地传输训练模型到服务器：**  
   `scp -r TRO_SRL_v3.2.0_28-16-34-54 QH-MAIS:/nfs/IGEvolution/IGEvolution/isaacgymenvs/runs/`  
   `scp -r /home/zdh232/mly/isaacgym/ MAIS10:/home/pc/mly/`  
   `scp -r /home/zdh232/mly/IGEvolution/ MAIS10:/home/pc/mly/`  

- **启动X11桌面共享：**  
   `x11vnc -display :1 -auth /run/user/1000/gdm/Xauthority -forever -rfbauth ~/.vnc/passwd -listen 0.0.0.0`  

### 0.1 一键配置Titan Ubuntu训练终端
```bash
conda activate rlgpu  
cd isaacgymenvs  
export LD_LIBRARY_PATH=/home/zdh/anaconda3/envs/rlgpu/lib  
export WANDB_API_KEY=d9e147c0c0f29ad02ca38e65742ce8ce2bbd52ab  
```
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

pytorch 1.13.1 cuda11.7   
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117` 

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

## 3. Humanoid Model Pretrain

使用预训练仿真模型（Human+Mini SRL `hsrl_mode1_v3_s1.xml`）进行Humanoid Pretrain。

训练：  
`cd isaacgymenvs`    
`python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 wandb_project=SRL_Evo experiment=AMP_Pretrain task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml' headless=True wandb_activate=True max_iterations=4000  sim_device=cuda:0 rl_device=cuda:0 num_envs=4096 `    

查看AMP示例结果：  
`python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 test=True task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml'  num_envs=4 checkpoint=saved_runs/AMP_HumanoidPretrain/network/AMP_HumanoidPretrain_24-16-28-38.pth sim_device=cuda:0 rl_device=cuda:0` 

`python SRL_Evo_train.py task=HumanoidAMPSRLGym_marl test=True task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml'  num_envs=4 train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_10-15-03-25/nn/AMP_Pretrain_10-15-03-34.pth   sim_device=cuda:0 rl_device=cuda:0` 








