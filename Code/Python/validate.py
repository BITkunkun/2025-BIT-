"""
final_validate.py - 验证传统CBPNG和基于BP神经网络的NNCBPNG性能对比

本文件实现论文中描述的两种制导律性能对比实验：
1. 传统解析式偏置比例导引(CBPNG)
2. 基于BP神经网络的偏置比例导引(NNCBPNG)

主要功能：
1. 加载训练好的BP神经网络模型
2. 对三种不同终端交会角进行仿真
3. 比较两种方法的终端落角和脱靶量
4. 绘制弹道倾角变化曲线
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体和负号显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False 

# 统一参数设置(三种情况共用参数)
N = 3                           # 导航系数(论文中设置为3)
v = 300                         # 导弹速度 (m/s)
r0 = 10000                      # 初始弹目距离 (m)
theta0_deg = 10                 # 初始弹道倾角 (°)
q0 = 0                          # 初始视线角 (rad)，目标在正前方
x0 = 0                          # 初始x坐标
h0 = 0                          # 初始高度

# 三种期望终端交会角(对应论文中的实验)
qr_deg_values = [-40, -60, -80] 
num_qr = len(qr_deg_values)

# 结果存储类
class Result:
    """存储仿真结果的类"""
    def __init__(self):
        self.b = None              # 偏置项b值
        self.terminal_angle = None # 终端弹道倾角(rad)
        self.end_distance = None   # 脱靶量(m)
        self.tt = None             # 时间序列
        self.theta_values = None   # 弹道倾角变化序列
        self.terminal_angle_deg = None # 终端弹道倾角(deg)

# 初始化结果存储对象
results_cbpng = [Result() for _ in range(num_qr)]  # 传统CBPNG结果
results_nncbpng = [Result() for _ in range(num_qr)] # NNCBPNG结果

# 加载训练好的PyTorch模型和标准化参数
model_path = "guidance_model_multiLayers_pytorch_3w.pth"
model_data = torch.load(model_path)

# 定义与训练时相同的网络结构
class MultiLayerNetwork(torch.nn.Module):
    """3层BP神经网络(与训练时结构一致)"""
    def __init__(self, input_size=4, hidden_size=15, output_size=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid激活函数
        
    def forward(self, x):
        """前向传播"""
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化模型并加载训练好的参数
model = MultiLayerNetwork()
model.load_state_dict(model_data['model_state_dict'])
model.eval()  # 设置为评估模式

# 获取标准化参数(与训练时一致)
input_mean = model_data['input_mean']
input_std = model_data['input_std']
output_mean = model_data['output_mean']
output_std = model_data['output_std']

def simulate_missile_flight(b, N, v, r0, q0, theta0, x0, h0):
    """
    导弹飞行仿真函数
    
    参数:
        b: 偏置项 (rad/s)
        N: 导航比
        v: 导弹速度 (m/s)
        r0: 初始距离 (m)
        q0: 初始视线角 (rad)
        theta0: 初始弹道倾角 (rad)
        x0: 初始x坐标 (m)
        h0: 初始高度 (m)
        
    返回:
        terminal_angle: 终端弹道倾角 (rad)
        miss_distance: 脱靶量 (m)
        t: 时间序列
        theta_values: 弹道倾角变化序列
    """
    def dynamics(t, y):
        """导弹运动微分方程"""
        r, q, theta, x, h = y
        eta = theta - q  # 速度矢量与视线夹角
        
        # 视线角变化率
        q_dot = -v * np.sin(eta) / r  
        # 弹道倾角变化率(比例导引+偏置项)
        theta_dot = N * q_dot + b  
        # 距离变化率
        r_dot = -v * np.cos(eta)  
        # x坐标变化率
        x_dot = v * np.cos(theta)  
        # 高度变化率
        h_dot = v * np.sin(theta)  
        return [r_dot, q_dot, theta_dot, x_dot, h_dot]
    
    def stop_event(t, y):
        """积分停止条件: 当高度小于0时停止仿真"""
        return y[4]  
    stop_event.terminal = True
    stop_event.direction = -1
    
    # 初始状态 [距离, 视线角, 弹道倾角, x坐标, 高度]
    y0 = [r0, q0, theta0, x0, h0]
    t_span = [0, 100]  # 最大仿真时间(s)
    
    # 求解微分方程
    sol = solve_ivp(dynamics, t_span, y0, events=stop_event, rtol=1e-6)
    
    # 提取结果
    terminal_angle = sol.y[2, -1]  # 终端弹道倾角
    theta_values = sol.y[2, :]     # 弹道倾角变化序列
    # 计算脱靶量(目标位置x=10000m, h=0m)
    miss_distance = np.sqrt((sol.y[3, -1] - 10000)**2 + sol.y[4, -1]**2)
    
    return terminal_angle, miss_distance, sol.t, theta_values

# 对三种终端交会角进行仿真
for i in range(num_qr):
    qr_deg = qr_deg_values[i]
    
    # 转换为弧度
    theta0 = np.deg2rad(theta0_deg)
    qr = np.deg2rad(qr_deg)
    theta_f = q0 + qr  # 期望终端弹道倾角
    
    # 1. 传统CBPNG理论b值计算(解析式)
    t_go = r0 / v  # 预计飞行时间
    b_theory = (N*q0 - theta0 - (N-1)*theta_f) / t_go
    
    # 2. 神经网络预测b值
    input_nn = torch.tensor([[r0, N, theta0_deg, qr_deg]], dtype=torch.float32)
    # 输入标准化(与训练时一致)
    normalized_input = (input_nn - input_mean) / input_std
    with torch.no_grad():
        normalized_output = model(normalized_input)
        # 输出反标准化
        b_pred_nn = normalized_output * output_std + output_mean
        b_pred_nn = b_pred_nn.item()  # 转换为标量
    
    print(f'CBPNG (qr = {qr_deg_values[i]}°) 公式b值：{b_theory:.6f}')
    print(f'NNCBPNG (qr = {qr_deg_values[i]}°) 预测b值：{b_pred_nn:.6f}')

    # 3. 理论b值仿真
    (results_cbpng[i].terminal_angle, 
     results_cbpng[i].end_distance, 
     results_cbpng[i].tt, 
     results_cbpng[i].theta_values) = simulate_missile_flight(
         b_theory, N, v, r0, q0, theta0, x0, h0)
    results_cbpng[i].b = b_theory
    
    # 4. 神经网络b值仿真
    (results_nncbpng[i].terminal_angle, 
     results_nncbpng[i].end_distance, 
     results_nncbpng[i].tt, 
     results_nncbpng[i].theta_values) = simulate_missile_flight(
         b_pred_nn, N, v, r0, q0, theta0, x0, h0)
    results_nncbpng[i].b = b_pred_nn

# 转换为度数方便显示
for i in range(num_qr):
    results_cbpng[i].terminal_angle_deg = np.rad2deg(results_cbpng[i].terminal_angle)
    results_nncbpng[i].terminal_angle_deg = np.rad2deg(results_nncbpng[i].terminal_angle)

# 打印终端落角对比结果
print('==================== 终端落角对比 ====================')
for i in range(num_qr):
    print(f'CBPNG (qr = {qr_deg_values[i]}°) 落角：{results_cbpng[i].terminal_angle_deg:.4f}°')
    print(f'NNCBPNG (qr = {qr_deg_values[i]}°) 落角：{results_nncbpng[i].terminal_angle_deg:.4f}°')

# 打印脱靶量对比结果
print('\n==================== 脱靶量对比 ====================')
for i in range(num_qr):
    print(f'CBPNG (qr = {qr_deg_values[i]}°) 脱靶量：{results_cbpng[i].end_distance:.6f} m')
    print(f'NNCBPNG (qr = {qr_deg_values[i]}°) 脱靶量：{results_nncbpng[i].end_distance:.6f} m')

# 绘制弹道倾角θ变化曲线
plt.figure(figsize=(8, 6), facecolor='white')
# 颜色和线型设置
colors = [(0.2, 0.4, 0.8), (0.6, 0.8, 1.0), (0.8, 0.2, 0.2), 
          (1.0, 0.6, 0.6), (0.1, 0.6, 0.1), (0.6, 0.9, 0.6)]
linestyles = ['--', '-']  # CBPNG用虚线，NNCBPNG用实线
labels = ['CBPNG', 'NNCBPNG']
line_width = 1

# 绘制每种情况的曲线
curve_index = 0
for i in range(num_qr):
    # 绘制CBPNG曲线
    linestyle_index = curve_index % 2
    plt.plot(results_cbpng[i].tt, np.rad2deg(results_cbpng[i].theta_values),
             color=colors[curve_index], linestyle=linestyles[linestyle_index], 
             linewidth=line_width,
             label=f'{labels[0]}, qr = {qr_deg_values[i]}°')
    curve_index += 1

    # 绘制NNCBPNG曲线
    linestyle_index = curve_index % 2
    plt.plot(results_nncbpng[i].tt, np.rad2deg(results_nncbpng[i].theta_values),
             color=colors[curve_index], linestyle=linestyles[linestyle_index], 
             linewidth=line_width,
             label=f'{labels[1]}, qr = {qr_deg_values[i]}°')
    curve_index += 1

# 图表装饰
plt.xlabel('时间 (s)')
plt.ylabel('弹道倾角 θ (°)')
plt.title('三种情况下弹道倾角θ变化曲线')
plt.legend(loc='best')
plt.grid(True)
plt.show()
