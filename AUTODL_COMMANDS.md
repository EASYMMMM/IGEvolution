# AutoDL 常用命令

## 登录与项目环境

以下登录命令在本地电脑执行：

```bash
ssh autodl-srl
```

以下命令在 AutoDL 服务器执行：

```bash
cd ~/IGEvolution/isaacgymenvs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srlgym
```

检查当前环境：

```bash
which python
conda env list
nvidia-smi
```

## tmux（会话名：srl）

```bash
# 有会话就连接，没有就创建
tmux new -A -s srl

# 查看会话
tmux ls

# 重新连接；-d 会断开旧终端连接
tmux attach -d -t srl

# 删除会话（会终止其中运行的程序）
tmux kill-session -t srl
```

快捷键：

```text
Ctrl+B，松开，再按 D    离开会话，程序继续运行
Ctrl+B，松开，再按 C    新建窗口
Ctrl+B，松开，再按 N/P  切换下一个/上一个窗口
Ctrl+B，松开，再按 %    左右分屏
Ctrl+B，松开，再按 "    上下分屏
```

> SSH 断开不会停止 tmux；实例关机或重启后 tmux 会话消失。

## 数据盘与 runs

```bash
cd ~/IGEvolution/isaacgymenvs

# 检查 runs 的真实位置
ls -ld runs
readlink -f runs

# 查看数据盘容量
df -h /root/autodl-tmp
du -sh runs

# 查看最近的实验目录
ls -laht runs | head -20
```

`runs` 应指向：

```text
/root/autodl-tmp/IGEvolution/runs
```

首次建立软链接（仅在 `runs` 不存在时执行）：

```bash
mkdir -p /root/autodl-tmp/IGEvolution/{runs,logs,videos,wandb}
ln -s /root/autodl-tmp/IGEvolution/runs runs
```

## 查找 checkpoint

```bash
# 所有 checkpoint
find -L runs -type f -name '*.pth'

# 最近生成的 20 个 checkpoint
find -L runs -type f -name '*.pth' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' | sort -r | head -20

# 某个实验的 checkpoint
ls -laht runs/实验目录/nn/
```

加载 checkpoint：

```bash
python SRL_Evo_train.py ... checkpoint=runs/实验目录/nn/模型.pth
```

通常 `实验名.pth` 是训练期间的最佳模型，`last_*.pth` 是最后一次保存的模型。

## 训练与日志

```bash
mkdir -p /root/autodl-tmp/IGEvolution/logs
export WANDB_DIR=/root/autodl-tmp/IGEvolution/wandb
export PYTHONUNBUFFERED=1
set -o pipefail

python -u SRL_Evo_train.py ... 2>&1 | \
  tee -a /root/autodl-tmp/IGEvolution/logs/训练名称.log
```

查看日志：

```bash
tail -f /root/autodl-tmp/IGEvolution/logs/训练名称.log
tail -n 200 /root/autodl-tmp/IGEvolution/logs/训练名称.log
```

检查训练进程：

```bash
pgrep -af SRL_Evo_train.py
nvidia-smi
```

## 训练结束自动关机

仅训练成功后关机：

```bash
set -o pipefail
python -u SRL_Evo_train.py ... 2>&1 | \
  tee -a /root/autodl-tmp/IGEvolution/logs/训练名称.log && \
  sync && /usr/bin/shutdown
```

无论训练成功或报错都关机：

```bash
python -u SRL_Evo_train.py ... 2>&1 | \
  tee -a /root/autodl-tmp/IGEvolution/logs/训练名称.log
sync
/usr/bin/shutdown
```

## TensorBoard 与视频

```bash
# 启动后通过 AutoDL 的 6006 自定义服务访问
tensorboard --logdir runs --host 0.0.0.0 --port 6006

# 查找录制的视频
find -L videos -type f -name '*.mp4' -exec ls -lh {} \;
```

## 下载结果到本地

以下命令在本地电脑执行：

```bash
# 下载单个 checkpoint
scp autodl-srl:/root/autodl-tmp/IGEvolution/runs/实验目录/nn/模型.pth .

# 下载整个实验目录
scp -r autodl-srl:/root/autodl-tmp/IGEvolution/runs/实验目录 .
```
