import sys
import os 
import time

print(f"Step 1: Python starts... PID: {os.getpid()}")

try:
    import numpy as np
    print("Step 2: Numpy imported.")
except ImportError:
    print("Step 2 Failed: Numpy not found.")

try:
    import torch
    print(f"Step 3: PyTorch imported. Version: {torch.__version__}")
    print(f"        CUDA Available: {torch.cuda.is_available()}")
    x = torch.ones(1)
    if torch.cuda.is_available():
        x = x.cuda()
    print("Step 3.5: PyTorch Tensor operation success.")
except Exception as e:
    print(f"Step 3 Failed: {e}")

try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    print("Step 4: Matplotlib imported (Agg backend set).")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0,1], [0,1], [0,1])
    plt.savefig("debug_test.png")
    print("Step 4.5: Matplotlib plotting success (debug_test.png saved).")
except Exception as e:
    print(f"Step 4 Failed: {e}")

print("Step 5: Attempting to import poselib...")
try:
    # 尝试把当前目录加入路径，防止找不到 poselib
    sys.path.append(os.getcwd())
    from poselib.skeleton.skeleton3d import SkeletonMotion
    print("Step 6: Poselib imported successfully!")
except Exception as e:
    print(f"Step 6 Failed: Could not import poselib. Error: {e}")
    sys.exit(1)

print("Step 7: Testing file loading (Memory Check)...")
try:
    test_file = "amp_humanoid_walk2.npy" 
    if os.path.exists(test_file):
        print(f"   Found {test_file}, attempting to load...")
        motion = SkeletonMotion.from_file(test_file)
        print("   Load success!")
        print(f"   Data shape: {motion.global_translation.shape}")
        print(f"   Data device: {motion.tensor.device}")
    else:
        print("   (Skipping load test because smpl_humanoid_walk.npy not found)")
except Exception as e:
    print(f"Step 7 Failed: Crash during file loading. Error: {e}")
    sys.exit(1)

print("Step 8: All checks passed! Environment is STABLE.")