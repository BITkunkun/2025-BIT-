import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import scipy.io as sio

def compute_theta_error(b, N, v, r0, q0, theta0, theta_f):
    """计算弹道倾角误差函数"""
    def dynamics(t, y):
        """导弹运动微分方程"""
        r, q, theta = y
        eta = theta - q
        
        q_dot = -v * np.sin(eta) / r
        theta_dot = N * q_dot + b
        r_dot = -v * np.cos(eta)
        return [r_dot, q_dot, theta_dot]
    
    def stop_event(t, y):
        """积分停止条件: 距离小于1米"""
        return y[0] - 1
    
    stop_event.terminal = True
    stop_event.direction = -1
    
    y0 = [r0, q0, theta0]
    t_span = (0, 100)  # 最大仿真时间
    
    try:
        sol = solve_ivp(dynamics, t_span, y0, events=stop_event, 
                        rtol=1e-6, atol=1e-8)
        if not sol.t_events[0].size:  # 没有触发停止事件
            return np.inf
        theta_real = sol.y[2, -1]  # 最终弹道倾角
        return theta_real - theta_f
    except:
        return np.inf

def main():
    # 参数定义
    v = 300  # 导弹速度 (m/s)
    
    # 生成参数范围
    r0_part1 = np.arange(5000, 7501, 100)    # 5000-7500m, 步长100m
    r0_part2 = np.arange(7550, 10001, 50)    # 7500-10000m, 步长50m
    r0 = np.concatenate((r0_part1, r0_part2))
    N_values = np.array([2, 3, 4])           # N ∈ {2,3,4}
    theta0 = np.arange(10, 21)               # θ0 ∈ [10,20]°
    qr = np.arange(-70, -39.5, 0.5)          # qr ∈ [-70,-40]°
    
    # 生成样本
    num_samples = 30000 # 样本个数
    data = np.zeros((num_samples, 4))        # 输入: [r0, N, θ0, qr]
    b_actual = np.zeros(num_samples)         # 输出: 偏置项b
    
    np.random.seed(42)  # 固定随机种子保证可重复性
    
    for i in range(num_samples):
        # 1. 生成四维参数(均匀采样)
        data[i, 0] = np.random.choice(r0)        # r0 ∈ [5,10] km
        data[i, 1] = np.random.choice(N_values)  # N ∈ {2,3,4}
        data[i, 2] = np.random.choice(theta0)    # θ0 ∈ [10,20]°
        data[i, 3] = np.random.choice(qr)       # qr ∈ [-70,-40]°
        
        # 2. 计算相关参数
        r0_val = data[i, 0]                     # 单位: m
        N = data[i, 1]
        theta0_val = np.deg2rad(data[i, 2])     # 单位: rad
        qr_val = np.deg2rad(data[i, 3])         # 单位: rad
        
        # 计算终端视线角
        q0 = 0                                  # 初始视线角为0°
        qf = q0 + qr_val                       # 终端视线角
        theta_f = qf                           # 终端弹道倾角θf = qf
        
        # 3. 理论计算偏置项(初始猜测值)
        t_go = r0_val / v                      # 飞行时间估计
        b_guess = (N*q0 - theta0_val - (N-1)*theta_f) / t_go
        
        # 4. 优化求解实际b值
        # 修改后的优化部分代码
        try:
            sol = root_scalar(
                lambda b: compute_theta_error(b, N, v, r0_val, q0, theta0_val, theta_f),
                x0=b_guess,
                x1=b_guess * 1.001,  # 提供第二个初始猜测点
                method='secant',       # 使用割线法
                xtol=1e-6,
                maxiter=100
            )
            b_actual[i] = sol.root
        except:
            b_actual[i] = b_guess
            print(f'优化失败样本{i}, 使用理论值b={b_guess:.4f}')
    
    # 保存数据集
    inputs = data                              # [r0(m), N, θ0(deg), qr(deg)]
    outputs = b_actual.reshape(-1, 1)          # 偏置项b(rad/s)
        
    # 如果需要PyTorch训练
    import torch
    torch.save({
        'inputs': torch.from_numpy(inputs).float(),
        'outputs': torch.from_numpy(outputs).float()
    }, 'guidance_dataset_python_3w.pt')

if __name__ == '__main__':
    main()
