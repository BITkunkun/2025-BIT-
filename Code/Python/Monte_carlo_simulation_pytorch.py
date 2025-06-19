"""
Monte_carlo_simulation_pytorch.py - 蒙特卡洛仿真验证

本文件实现论文中描述的蒙特卡洛仿真实验，用于验证：
1. 基于BP神经网络的NNCBPNG在不同初始条件下的性能
2. 统计终端落角误差分布
3. 分析方法的鲁棒性和稳定性

主要功能：
1. 随机生成初始条件(距离、弹道倾角、终端交会角)
2. 进行500次蒙特卡洛仿真
3. 计算性能指标
4. 绘制误差分布图
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# 设置中文字体(用于图表显示中文)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False 

class MultiLayerNetwork(nn.Module):
    """3层BP神经网络(与训练时结构一致)
    网络结构: 输入层(4) -> 隐藏层1(15) -> 隐藏层2(15) -> 隐藏层3(15) -> 输出层(1)
    使用Sigmoid激活函数和Xavier初始化
    """
    def __init__(self, input_size, hidden_size=15, output_size=1):
        super(MultiLayerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid3 = nn.Sigmoid()
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # 初始化
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.fc3(x)
        x = self.sigmoid3(x)
        x = self.fc4(x)
        return x

def main():
    """主函数 - 执行蒙特卡洛仿真流程"""
    # 仿真参数设置(与论文一致)
    # num_samples: 蒙特卡洛仿真次数(500次)
    # v: 导弹速度(300m/s)
    # N: 导航比(3)
    # target_x_range: 目标初始x坐标范围(5-10km)
    # theta0_range: 初始弹道倾角范围(10-20°)
    # theta_f_range: 期望终端交会角范围(-90°到-50°)
    num_samples = 500
    v = 300
    N = 3
    missile_pos = np.array([0, 0])
    target_x_range = [5000, 10000]
    theta0_range = [10, 20]
    theta_f_range = [-90, -50]

    # CPU进行计算
    device = torch.device("cpu")
    print("Using device: cpu (forced for compatibility)")

    # 加载模型和参数
    checkpoint = torch.load('guidance_model_multiLayers_pytorch_3w.pth', map_location=device, weights_only=False)
    model = MultiLayerNetwork(4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 标准化参数
    input_mean = checkpoint['input_mean'].to(device)
    input_std = checkpoint['input_std'].to(device)
    output_mean = checkpoint['output_mean'].to(device)
    output_std = checkpoint['output_std'].to(device)

    # 初始化存储
    errors = np.zeros(num_samples)

    # 生成随机参数
    target_x = np.random.uniform(*target_x_range, num_samples)
    theta0 = np.random.uniform(*theta0_range, num_samples)
    theta_f_desired = np.random.uniform(*theta_f_range, num_samples)

    # 蒙特卡洛仿真
    for i in range(num_samples):
        # 目标初始位置
        target_pos = np.array([target_x[i], 0])
        
        # 计算初始距离
        r0 = np.linalg.norm(missile_pos - target_pos)
        
        # 神经网络输入
        input_tensor = torch.tensor([r0, N, theta0[i], theta_f_desired[i]], 
                                   dtype=torch.float32, device=device)
        
        # 标准化并预测
        input_normalized = (input_tensor - input_mean) / input_std
        with torch.no_grad():
            b_pred = (model(input_normalized) * output_std + output_mean).item()

        # 运动学仿真
        theta_actual, miss = simulate_missile_flight(
            float(b_pred), N, v, r0, theta0[i], target_pos
        )
        errors[i] = abs(theta_actual - theta_f_desired[i])

    # 统计结果
    max_err = np.max(errors)
    min_err = np.min(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)

    print('=============== 蒙特卡洛仿真结果 ===============')
    print(f'最大误差: {max_err:.4f}°')
    print(f'最小误差: {min_err:.4f}°')
    print(f'平均误差: {mean_err:.4f}°')
    print(f'标准误差: {std_err:.4f}°')

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_samples), errors, alpha=0.6, edgecolors='w')
    plt.xlabel('样本索引')
    plt.ylabel('角度误差 (°)')
    plt.title('多层神经网络制导误差分布')
    plt.grid(True)
    plt.show()

def simulate_missile_flight(b, N, v, r0, theta0_deg, target_pos):
    """导弹飞行仿真函数
    参数:
        b: 偏置项(rad/s)
        N: 导航比
        v: 导弹速度(m/s)
        r0: 初始距离(m)
        theta0_deg: 初始弹道倾角(度)
        target_pos: 目标位置(x,h)(m)
    返回:
        theta_final: 终端弹道倾角(度)
        miss_distance: 脱靶量(m)
    """
    # 初始条件转换
    theta0_rad = np.deg2rad(theta0_deg)
    y0 = [r0, 0, theta0_rad, 0, 0]  # [r, q, theta, x, h]

    # 定义动力学方程
    def dynamics(t, y):
        r, q, theta, x, h = y
        eta = theta - q
        return [
            -v * np.cos(eta),          # r_dot
            -v * np.sin(eta)/r,        # q_dot
            N * (-v * np.sin(eta)/r) + b,  # theta_dot
            v * np.cos(theta),         # x_dot
            v * np.sin(theta)          # h_dot
        ]

    # 终止条件
    def stop_condition(t, y):
        return y[0] - 1
    stop_condition.terminal = True

    # 数值求解
    sol = solve_ivp(dynamics, [0, 1000], y0, events=stop_condition, rtol=1e-6)
    
    # 计算最终结果
    theta_final = np.rad2deg(sol.y[2, -1])
    miss_distance = np.linalg.norm(sol.y[3:, -1] - target_pos)
    
    return theta_final, miss_distance

if __name__ == '__main__':
    main()
