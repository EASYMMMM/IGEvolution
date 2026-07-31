# AutoDL 环境安装说明

本文档用于在 AutoDL 的 RTX 5090 实例上安装本项目运行环境。核心原则是：

- 环境、Isaac Gym、项目代码放在 `/root`，方便后续打包进镜像。
- 训练输出、checkpoint、wandb、临时数据放在 `/root/autodl-tmp`，避免系统盘爆掉。
- PyTorch 不通过 `setup.py` 或普通 `requirements` 自动安装，必须手动安装固定版本 wheel。
- Isaac Gym 和 IGEvolution 都用 editable install，但建议加 `--no-deps`，避免 pip 自动改掉 torch。

## 1. 需要准备的文件

需要提前准备并上传到 AutoDL：

```text
isaacgym.tar.gz
torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
IGEvolution 项目代码
```

推荐先把两个本地文件上传到 AutoDL 数据盘：

```text
/root/autodl-tmp/isaacgym.tar.gz
/root/autodl-tmp/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
```

如果从本地电脑上传，示例：

```bash
scp -P <AutoDL端口> /path/to/isaacgym.tar.gz root@<AutoDL地址>:/root/autodl-tmp/
scp -P <AutoDL端口> /path/to/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl root@<AutoDL地址>:/root/autodl-tmp/
```

## 2. 整理目录

登录 AutoDL 后执行：

```bash
cd /root

mkdir -p /root/wheels
cp /root/autodl-tmp/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl /root/wheels/

tar -xzf /root/autodl-tmp/isaacgym.tar.gz -C /root
```

检查 Isaac Gym 解压后的目录：

```bash
ls -lh /root
```

如果解压出来的目录名不是 `/root/isaacgym`，重命名：

```bash
mv /root/<解压出来的目录名> /root/isaacgym
```

最终建议目录结构：

```text
/root/isaacgym
/root/IGEvolution
/root/wheels/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
/root/autodl-tmp
```

## 3. 创建 Conda 环境

如果 `conda activate` 报错 `Run 'conda init' before 'conda activate'`，先执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

创建环境：

```bash
conda create -n srlgym python=3.8 -y
conda activate srlgym
```

可选：安装 CUDA runtime 相关库。

这一步不是绝对必需。如果 AutoDL 上 conda 源超时，或者后面安装 torch wheel 后已经能通过 CUDA 验证，可以跳过这一条。torch wheel 本身会安装/携带一批运行所需的 CUDA 依赖。

```bash
conda install -c nvidia cuda-runtime cuda-libraries cuda-nvrtc cuda-nvtx cuda-cupti cudnn -y
```

设置动态库路径。

即使跳过上面的 conda CUDA runtime 安装，也建议保留这行。Isaac Gym 的二进制库比较依赖 conda 环境里的动态库路径。

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

更推荐把它写成 conda 环境的自动激活脚本，这样每次 `conda activate srlgym` 后都会自动生效：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

写入后重新激活一次环境：

```bash
conda deactivate
conda activate srlgym
```

## 4. 安装 PyTorch

必须先安装本地 torch wheel：

```bash
pip install /root/wheels/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
```

安装 torchvision 和 torchaudio 时不要让 pip 自动替换 torch：

```bash
pip install torchvision==0.18.1 torchaudio==2.3.1 --no-deps
```

验证 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期望看到：

```text
2.3.0a0+git63d5e92 12.8
True
NVIDIA GeForce RTX 5090
```

如果这里验证通过，就可以继续安装 Isaac Gym。即使第 3 节的 `conda install -c nvidia ...` 因网络问题失败，也不需要先卡在那一步。

## 5. 安装 Isaac Gym

先安装 Isaac Gym 需要的基础依赖：

```bash
pip install numpy==1.23.5 scipy pyyaml pillow imageio ninja
```

然后安装 Isaac Gym：

```bash
cd /root/isaacgym/python
pip install -e . --no-deps
```

验证 Isaac Gym：

```bash
cd /root/isaacgym/python/examples
python joint_monkey.py --headless
```

如果该命令能正常启动并退出，说明 Isaac Gym 基本可用。

## 6. 下载 IGEvolution

推荐把项目放到 `/root/IGEvolution`，这样可以打包进 AutoDL 镜像。

```bash
cd /root
git clone --depth 1 https://github.com/EASYMMMM/IGEvolution.git
```

如果 AutoDL 网络不稳定，普通 clone 报 `RPC failed`、`early EOF`，可以改用：

```bash
cd /root
git clone --depth 1 --filter=blob:none https://github.com/EASYMMMM/IGEvolution.git
```

如果仍然失败，可以在本地把项目打包成不带 `.git` 的压缩包上传到 AutoDL。

## 7. 安装 IGEvolution 依赖

进入环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate srlgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

安装项目依赖。注意这里不安装 torch：

```bash
pip install gym==0.23.1 omegaconf termcolor jinja2 hydra-core==1.3.2 rl-games==1.6.1 pyvirtualdisplay urdfpy==0.0.22 pysdf==0.1.9 warp-lang==0.10.1 trimesh==3.23.5 matplotlib
```

安装项目本身：

```bash
cd /root/IGEvolution
pip install -e . --no-deps
```

这里使用 `pip install -e .` 是为了让 `isaacgymenvs` 作为可编辑包安装。之后修改代码无需重新安装。

## 8. 项目验证

先跑一个基础任务：

```bash
cd /root/IGEvolution/isaacgymenvs
python train.py task=Cartpole headless=True sim_device=cuda:0 rl_device=cuda:0
```

再跑项目任务时，建议显式指定设备：

```bash
python SRL_Evo_train.py task=SRLBot headless=True sim_device=cuda:0 rl_device=cuda:0
```

## 9. 训练输出放到数据盘

不要把训练输出长期放在 `/root/IGEvolution/isaacgymenvs/runs`。建议链接到数据盘：

```bash
mkdir -p /root/autodl-tmp/IGEvolution_runs
cd /root/IGEvolution/isaacgymenvs

if [ -e runs ] && [ ! -L runs ]; then
  mv runs /root/autodl-tmp/IGEvolution_runs_backup_$(date +%Y%m%d_%H%M%S)
fi

ln -s /root/autodl-tmp/IGEvolution_runs runs
```

wandb 也建议放到数据盘：

```bash
mkdir -p /root/autodl-tmp/wandb
export WANDB_DIR=/root/autodl-tmp/wandb
```

## 10. 常用启动模板

每次新开终端：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate srlgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /root/IGEvolution/isaacgymenvs
```

示例训练：

```bash
python SRL_Evo_train.py task=SRLBot headless=True sim_device=cuda:0 rl_device=cuda:0
```

## 11. 常见问题

### conda activate 报错

执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate srlgym
```

### git clone 中断

使用浅克隆：

```bash
git clone --depth 1 https://github.com/EASYMMMM/IGEvolution.git
```

或者 partial clone：

```bash
git clone --depth 1 --filter=blob:none https://github.com/EASYMMMM/IGEvolution.git
```

### pip install -e . 自动安装了错误版本 torch

本项目不应该依赖 `setup.py` 自动安装 torch。正确做法是：

```bash
pip install /root/wheels/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
cd /root/IGEvolution
pip install -e . --no-deps
```

### ImportError: libpython 或动态库找不到

典型报错：

```text
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

确认：

```bash
echo $CONDA_PREFIX
echo $LD_LIBRARY_PATH
find $CONDA_PREFIX -name 'libpython3.8.so.1.0'
```

然后重新设置：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

也可以永久写入当前 conda 环境：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate
conda activate srlgym
```

如果 PyTorch 已经能识别 RTX 5090，但 Isaac Gym 报动态库相关错误，优先检查这一行是否在当前终端生效。

### conda 安装 CUDA runtime 超时

如果出现类似：

```text
CondaHTTPError: HTTP 000 CONNECTION FAILED
ReadTimeoutError
```

可以先跳过 conda CUDA runtime 安装，直接安装 torch wheel：

```bash
pip install /root/wheels/torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl
pip install torchvision==0.18.1 torchaudio==2.3.1 --no-deps
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果验证输出 `True` 和 `NVIDIA GeForce RTX 5090`，继续后面的 Isaac Gym 和 IGEvolution 安装即可。

### CUDA 可见性检查

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 12. 镜像打包建议

建议打包进 AutoDL 镜像的内容：

```text
/root/miniconda3/envs/srlgym
/root/isaacgym
/root/IGEvolution
/root/wheels
```

不建议打包进镜像的内容：

```text
/root/autodl-tmp/runs
/root/autodl-tmp/wandb
/root/autodl-tmp/checkpoints
```

一句话：环境和代码进镜像，训练结果留数据盘。
