% ======================================================================
% 本代码为论文《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）的复现尝试
% 引用时请注明原论文信息：
% 刘畅, 王江, 范世鹏, 等. 基于BP神经网络的自适应偏置比例导引[J]. 兵工学报, 2022, 43(11): 2798-2812.
% DOI: 10.12382/bgxb.2021.0594
% 
% ======================================================================
% 2025上半年“制导与控制”春季学期小组作业
% 组员：王宇轩、孙伟轩、常珈毓、罗嘉豪
% ======================================================================
%% 基于指数衰减模型的变偏置量导引律仿真模块
% 创新点：在传统常值偏置基础上，引入距离触发的指数衰减机制
% 参考论文思路：利用非线性映射优化偏置项，提升大范围机动精度

% 参数定义
v = 300; % 导弹速度 (m/s)
trigger_distance = 800; % b值开始变化的距离(m)
linear_points = 100; % 下降区间点数

% 生成参数范围（保持不变）
r0_part1 = 5000:100:7500;
r0_part2 = 7500+50:50:10000;
r0 = [r0_part1, r0_part2];
N_values = 2:1:4;
theta0 = 10:1:20;
qr = -70:0.5:-40;

% 生成样本
num_samples = 10000;
data = zeros(num_samples, 4);
b_actual_initial = zeros(num_samples, 1); % 初始b值
b_actual_linear = zeros(num_samples, linear_points); % 下降部分(800m→0m)

options = optimset('TolX', 1e-6, 'Display', 'off');

for i = 1:num_samples
    %% 参数采样（保持不变）
    data(i,1) = r0(randi(length(r0)));
    data(i,2) = N_values(randi(3));
    data(i,3) = theta0(randi(11));
    data(i,4) = qr(randi(length(qr)));
    
    %% 计算参数
    r0_val = data(i,1);
    N = data(i,2);
    theta0_val = deg2rad(data(i,3));
    qr_val = deg2rad(data(i,4));
    
    q0 = 0;
    qf = q0 + qr_val;
    theta_f = qf;
    
    %% 计算初始b值
    t_go = r0_val / v;
    b_guess = (N*q0 - theta0_val - (N-1)*theta_f) / t_go;
    
    try
        b_actual_initial(i) = fzero(@(b) compute_theta_error(b, N, v, r0_val, q0, theta0_val, theta_f), b_guess, options);
    catch
        b_actual_initial(i) = b_guess;
        fprintf('优化失败样本%d，使用理论值b=%.4f\n', i, b_guess);
    end
    
end

%% 合并结果
inputs = data;               % [r0(m), N, θ0(deg), qr(deg)]
outputs = b_actual_initial; 

%% 保存数据集
save('guidance_dataset_exp.mat', 'inputs', 'outputs', 'trigger_distance');

%% 修改后的误差计算函数（需记录距离信息）
function [error, recorded_distances, recorded_b] = compute_theta_error(b, N, v, r0, q0, theta0, theta_f)
    trigger_distance = 800;
    recorded_distances = [];
    recorded_b = [];
    
    function dydt = dynamics(~, y)
        r = y(1);
        q = y(2);
        theta = y(3);
        eta = theta - q;
        
        q_dot = -v * sin(eta) / r;
        % 记录当前距离和b值(exp变b)
        if r <= trigger_distance
            k = 5;
            decay_factor = exp(-k * (trigger_distance - r) / trigger_distance);
            b1 = b * decay_factor;
            theta_dot = N * q_dot + b1;
        else
            theta_dot = N * q_dot + b;
        end
        
        r_dot = -v * cos(eta);
        dydt = [r_dot; q_dot; theta_dot];
    end

    function [value, isterminal, direction] = stop_event(~, y)
        value = y(1) - 1;
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0];
    tspan = [0, 1000];
    options = odeset('Events', @stop_event, 'RelTol', 1e-6);
    
    try
        [~, y] = ode45(@dynamics, tspan, y0, options);
        if isempty(y)
            error = Inf;
        else
            theta_real = y(end, 3);
            error = theta_real - theta_f;
        end
    catch
        error = Inf;
    end
end
