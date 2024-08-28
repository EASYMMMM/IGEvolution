import random

def SRL_designer(self):
    # 基础尺寸
    base_params = {
        "first_leg_lenth" : 0.40,
        "first_leg_size"  : 0.03,
        "second_leg_lenth": 0.80,
        "second_leg_size" : 0.03,
        "third_leg_size"  : 0.03,
    }
    
    # 生成随机浮动的尺寸，并将结果保留到四位小数
    srl_params = {
        key: round(random.uniform(value * 0.7, value * 1.3), 4)  # 在上下浮动30%范围内生成随机值并保留四位小数
        for key, value in base_params.items()
    }
    
    return srl_params

# 测试函数
if __name__ == "__main__":
    # 创建类实例
    class Test:
        SRL_designer = SRL_designer
    
    # 实例化类
    test_instance = Test()
    
    # 运行设计函数
    random_params = test_instance.SRL_designer()
    print(random_params)