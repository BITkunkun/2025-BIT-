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

%% 基于BP神经网络的自适应偏置比例导引算法数据生成模块
% 参考论文：《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）
% 功能：生成用于训练BP神经网络的样本数据集，包含四维输入参数与对应的偏置项b
% 关键逻辑：基于论文中的灵敏度分析结果分层采样，通过动力学仿真求解真实偏置项
% 对应论文第2.3节"样本建立"

% 参数定义
v = 300; % 导弹速度 (m/s)
% 生成参数范围（基于分层采样策略）
r0_part1 = 5000:100:7500;
r0_part2 = 7500+50:50:10000;
r0 = [r0_part1, r0_part2];
N_values = 2:1:4;
theta0 = 10:1:20;
qr = -70:0.5:-40;

% 生成四维参数空间样本
num_samples = 10000;
data = zeros(num_samples, 4);     % 输入：[r0, N, θ0, qr]
b_actual = zeros(num_samples, 1);

% 优化配置
options = optimset('TolX', 1e-6, 'Display', 'off', 'MaxIter', 100);

for i = 1:num_samples
    % 生成四维参数（均匀采样）
    data(i,1) = r0(randi(length(r0)));
    data(i,2) = N_values(randi(3));
    data(i,3) = theta0(randi(11));
    data(i,4) = qr(randi(length(qr)));
    
    % 运动学参数转换
    r0_val = data(i,1);
    N = data(i,2);
    theta0_val = deg2rad(data(i,3));
    qr_val = deg2rad(data(i,4));
    
    % 计算终端视线角（论文定义qr = qf - q0）
    q0 = 0; 
    qf = q0 + qr_val;
    % 终端弹道倾角θf = qf（论文假设条件）
    theta_f = qf;
    
    % 解析法，理论计算偏置项
    t_go = r0_val / v;
    b_guess = (N*q0 - theta0_val - (N-1)*theta_f) / t_go;
    
    % 优化求解实际b值，利用fzero优化器求解使终端角度误差为0的b值
    try
        b_actual(i) = fzero(@(b) compute_theta_error(b, N, v, r0_val,...
                          q0, theta0_val, theta_f), b_guess, options);
    catch
        b_actual(i) = b_guess;
        fprintf('优化失败样本%d，使用理论值b=%.4f\n', i, b_guess);
    end
end

% 保存数据集
inputs = data;               % [r0(m), N, θ0(deg), qr(deg)]
outputs = b_actual;          % 偏置项b(rad/s)
save('guidance_dataset.mat', 'inputs', 'outputs');

% 误差计算函数
function error = compute_theta_error(b, N, v, r0, q0, theta0, theta_f)
    % 弹目相对运动动力学模型
    function dydt = dynamics(~, y)
        r = y(1);    % 弹目距离
        q = y(2);    % 视线角
        theta = y(3);% 弹道倾角
        eta = theta - q;
        
        q_dot = -v * sin(eta) / r;
        theta_dot = N * q_dot + b;
        r_dot = -v * cos(eta);
        dydt = [r_dot; q_dot; theta_dot];
    end

    function [value, isterminal, direction] = stop_event(~, y)
        value = y(1) - 1;  % 当距离小于1米时停止
        isterminal = 1;    % 停止积分
        direction = -1;    % 检测下降沿
    end
    
    % 初始状态与仿真配置
    y0 = [r0; q0; theta0];
    tspan = [0, 1000];     % 最大仿真时间
    options = odeset('Events', @stop_event, 'RelTol', 1e-6);
    
    % 数值积分与误差计算
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