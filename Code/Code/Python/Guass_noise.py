##############################################################################
"""
Guass_noise.py - 高斯白噪声仿真验证

本文件实现论文中描述的高斯白噪声条件下的制导律性能验证实验：
1. 在视线角变化率(q_dot)中加入高斯白噪声
2. 比较传统CBPNG和基于BP神经网络的NNCBPNG在噪声条件下的性能
3. 分析终端落角和脱靶量指标

主要功能：
1. 定义带噪声的导弹运动微分方程
2. 加载训练好的BP神经网络模型
3. 对三种不同终端交会角进行带噪声仿真
4. 比较两种方法的性能指标
5. 绘制弹道倾角变化曲线
"""
##############################################################################

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 结果存储类
class Result:
    """存储仿真结果的类"""
    def __init__(self):
        self.b = None              # 偏置项b值
        self.terminal_angle = None # 终端弹道倾角(rad)
        self.end_distance = None   # 脱靶量(m)
        self.time = None           # 时间序列
        self.theta_values = None   # 弹道倾角变化序列

# 定义与训练时相同的神经网络结构
class MultiLayerNetwork(nn.Module):
    """3层BP神经网络(与训练时结构一致)"""
    def __init__(self, input_size=4, hidden_size=15, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()  # 第一层激活函数
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()  # 第二层激活函数
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid3 = nn.Sigmoid()  # 第三层激活函数
        self.fc4 = nn.Linear(hidden_size, output_size)  # 输出层
        
        # Xavier初始化权重
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播"""
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.sigmoid3(self.fc3(x))
        return self.fc4(x)
    
# 全局变量用于存储随机数和当前索引
RandomNumbers = np.random.normal(0, (np.sqrt(0.6) * np.pi) / 180, 500)  # 500个正态分布随机数(标准差0.6°)
Index = 0  # 索引初始化

def simulate_missile_flight(b, N, v, r0, q0, theta0, target_pos=(10000, 0)):
    """
    带噪声的导弹飞行仿真函数
    
    参数:
        b: 偏置项 (rad/s)
        N: 导航比
        v: 导弹速度 (m/s)
        r0: 初始距离 (m)
        q0: 初始视线角 (rad)
        theta0: 初始弹道倾角 (rad)
        target_pos: 目标位置 (x, h) (m)
        
    返回:
        Result对象包含仿真结果
    """
    global RandomNumbers, Index
    
    def dynamics(t, y):
        """带噪声的导弹运动微分方程"""
        global RandomNumbers, Index
        r, q, theta, x, h = y
        if r < 1e-3:  # 避免除零错误
            return [0, 0, 0, 0, 0]
        
        eta = theta - q  # 速度矢量与视线夹角
        q_dot = -v * np.sin(eta) / r
        
        # 加入高斯白噪声(论文中标准差0.6°)
        q_dot += RandomNumbers[Index]
        
        # 更新索引(循环使用随机数)
        Index += 1
        if Index >= len(RandomNumbers):
            Index = 0
            
        theta_dot = N * q_dot + b  # 比例导引+偏置项
        r_dot = -v * np.cos(eta)   # 距离变化率
        x_dot = v * np.cos(theta)  # x坐标变化率
        h_dot = v * np.sin(theta)  # 高度变化率
        return [r_dot, q_dot, theta_dot, x_dot, h_dot]
    
    # 事件检测: 当高度h从正下降至零时触发
    def stop_condition(t, y):
        return y[4]
    stop_condition.terminal = True
    stop_condition.direction = -1
    
    # 初始状态: h=1e-3避免初始触发
    y0 = [r0, q0, theta0, 0, 1e-3]
    
    # 求解微分方程(提高精度)
    sol = solve_ivp(
        dynamics,
        [0, 300],  # 最大仿真时间300秒
        y0,
        events=stop_condition,
        rtol=1e-6,   # 相对容差
        atol=1e-6,   # 绝对容差
        dense_output=True
    )
    
    # 存储结果
    result = Result()
    result.b = b
    result.terminal_angle = sol.y[2, -1]  # 终端弹道倾角
    result.time = sol.t                   # 时间序列
    result.theta_values = sol.y[2, :]     # 弹道倾角变化
    
    # 计算脱靶量(欧几里得距离)
    missile_pos = (sol.y[3, -1], sol.y[4, -1])
    result.end_distance = np.hypot(
        missile_pos[0] - target_pos[0],
        missile_pos[1] - target_pos[1]
    )
    return result

def main():
    """主函数"""
    # 统一参数设置(与论文一致)
    N = 3           # 导航比
    v = 300         # 导弹速度(m/s)
    r0 = 10000      # 初始距离(m)
    theta0_deg = 10 # 初始弹道倾角(°)
    q0 = 0          # 初始视线角(rad)
    
    # 三种期望终端交会角(°)
    qr_deg_values = [-40, -60, -80]

    # 加载训练好的模型
    device = torch.device("cpu")
    checkpoint = torch.load('guidance_model_multiLayers_pytorch_3w.pth', map_location=device)
    model = MultiLayerNetwork(4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 评估模式

    # 获取标准化参数(与训练时一致)
    input_mean = checkpoint['input_mean'].cpu()
    input_std = checkpoint['input_std'].cpu()
    output_mean = checkpoint['output_mean'].cpu()
    output_std = checkpoint['output_std'].cpu()

    # 初始化结果存储
    results = {'CBPNG': [], 'NNCBPNG': []}

    # 对三种终端交会角进行仿真
    for qr_deg in qr_deg_values:
        # 转换为弧度
        theta0 = np.deg2rad(theta0_deg)
        qr = np.deg2rad(qr_deg)
        theta_f = q0 + qr  # 期望终端弹道倾角
        
        # 1. 传统CBPNG理论b值计算
        t_go = r0 / v  # 预计飞行时间
        b_theory = (N * q0 - theta0 - (N-1)*theta_f) / t_go
        
        # 2. 神经网络预测b值
        input_nn = torch.tensor([r0, N, theta0_deg, qr_deg], dtype=torch.float32)
        input_norm = (input_nn - input_mean) / input_std  # 输入标准化
        with torch.no_grad():
            b_pred = (model(input_norm) * output_std + output_mean).item()  # 输出反标准化
        
        print(f'CBPNG (qr = {qr_deg}°) 公式b值：{b_theory:.6f}')
        print(f'NNCBPNG (qr = {qr_deg}°) 预测b值：{b_pred:.6f}')

        # 3. 带噪声仿真
        results['CBPNG'].append(simulate_missile_flight(b_theory, N, v, r0, q0, theta0))
        results['NNCBPNG'].append(simulate_missile_flight(b_pred, N, v, r0, q0, theta0))

    # 打印终端落角对比结果
    print('================ 终端落角 ================')
    for i, qr in enumerate(qr_deg_values):
        cbpng = results['CBPNG'][i]
        nncbpng = results['NNCBPNG'][i]
        print(f'qr={qr}° | CBPNG: {np.rad2deg(cbpng.terminal_angle):.2f}° '
              f'vs NNCBPNG: {np.rad2deg(nncbpng.terminal_angle):.2f}°')

    # 打印脱靶量对比结果
    print('\n================ 脱靶量 ================')
    for i, qr in enumerate(qr_deg_values):
        cbpng = results['CBPNG'][i]
        nncbpng = results['NNCBPNG'][i]
        print(f'qr={qr}° | CBPNG: {cbpng.end_distance:.4f}m '
              f'vs NNCBPNG: {nncbpng.end_distance:.4f}m')

    # 绘制弹道倾角变化曲线
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    line_styles = ['--', '-', ':']   # 不同线型区分不同qr值
    handles = []  # 图例句柄

    # 绘制每种情况的曲线
    for i, (method, res_list) in enumerate(results.items()):
        for j, result in enumerate(res_list):
            time_array = result.time
            theta_values_array = np.rad2deg(result.theta_values)
            handle, = plt.plot(time_array, theta_values_array, 
                               color=colors[i], linestyle=line_styles[j],
                               label=f'{method} (qr={qr_deg_values[j]}°)')
            handles.append(handle)

    # 图表装饰
    plt.xlabel('时间 (s)')
    plt.ylabel('弹道倾角 (°)')
    plt.title('弹道倾角变化对比(带高斯白噪声)')
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
