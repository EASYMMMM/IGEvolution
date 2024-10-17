import torch
import time

print('Test Start')
# 设置设备为 GPU 0（cuda:0）
device = torch.device("cuda:0")

# 设置测试参数
num_batches = 100  # 批次数量
matrix_size = 20000  # 矩阵大小

# 初始化两个大矩阵
x = torch.rand(matrix_size, matrix_size, device=device)
y = torch.rand(matrix_size, matrix_size, device=device)

# 预热GPU，避免第一次调用的开销
torch.matmul(x, y)
torch.cuda.synchronize()

# 记录开始时间
start_time = time.time()

# 进行多次矩阵乘法运算
for _ in range(num_batches):
    print('Num batches:',_)
    result = torch.matmul(x, y)  # 矩阵乘法
    result = result.relu()  # 计算ReLU激活
    torch.cuda.synchronize()  # 确保每次运算完成后再进行下一步

# 记录结束时间
end_time = time.time()

# 计算总时间
total_time = end_time - start_time
print(f"总计算时间: {total_time:.4f} 秒")
print(f"每批次平均时间: {total_time / num_batches:.4f} 秒")
