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
%% 基于指数衰减模型的变偏置量导引律仿真模块（数据集生成准备）
% 创新点：在传统常值偏置基础上，引入距离触发的线性衰减机制
% 参考论文思路：利用非线性映射优化偏置项，提升大范围机动精度

% 参数定义
v = 300; % 导弹速度 (m/s)，根据论文设定
trigger_distance = 800; % b值开始线性下降的距离(m)
linear_points = 100; % 线性下降区间点数

% 生成参数范围
r0_part1 = 5000:100:7500;     % 5000-7500m，步长100m
r0_part2 = 7500+50:50:10000;  % 7500-10000m，步长50m
r0 = [r0_part1, r0_part2];
N_values = 2:1:4;
theta0 = 10:1:20;
qr = -70:0.5:-40;

% 生成样本
num_samples = 1000;
inputs = zeros(num_samples, 4);     % 输入：[r0, N, θ0, qr]
outputs = zeros(num_samples, 101);  % 输出：[b_const, b_linear1, ..., b_linear100]

% 优化配置
options = optimset('TolX', 1e-6, 'Display', 'off', 'MaxIter', 100);

for i = 1:num_samples
    %% 1. 生成四维参数（均匀采样）
    inputs(i,1) = r0(randi(length(r0)));        % r0 ∈ [5,10] km
    inputs(i,2) = N_values(randi(3));           % N ∈ {2,3,4}
    inputs(i,3) = theta0(randi(11));            % θ0 ∈ [10,20]°
    inputs(i,4) = qr(randi(length(qr)));        % qr ∈ [-70,-40]°
    
    %% 2. 参数计算
    r0_val = inputs(i,1);      % 单位：m
    N = inputs(i,2);
    theta0_val = deg2rad(inputs(i,3));  % 单位：rad
    qr_val = deg2rad(inputs(i,4));      % 单位：rad
    
    q0 = 0;                  % 假设初始视线角为0°
    qf = q0 + qr_val;        % 终端视线角
    theta_f = qf;            % 终端弹道倾角
    
    %% 3. 理论计算初始猜测值
    t_go = r0_val / v;
    b_guess = (N*q0 - theta0_val - (N-1)*theta_f) / t_go;
    
    %% 4. 优化求解b值（分两阶段）
    try
        % 第一阶段：优化恒定b值（距离>800m时）
        b_const = fzero(@(b) compute_error_constant_b(b, N, v, r0_val,...
                          q0, theta0_val, theta_f, trigger_distance),...
                          b_guess, options);
        
        % 第二阶段：优化线性下降起始值（确保终端角度准确）
        b_optimized = fzero(@(b) compute_error_variable_b(b, N, v, r0_val,...
                          q0, theta0_val, theta_f, trigger_distance),...
                          b_const, options);
        
        % 生成线性下降段（从800m到0m）
        distances = linspace(trigger_distance, 0, linear_points);
        b_linear = b_optimized * (distances / trigger_distance);
        
        % 存储结果
        outputs(i,:) = [b_optimized, b_linear];
        
    catch
        % 优化失败时使用理论值
        fprintf('优化失败样本%d，使用理论值\n', i);
        b_linear = b_guess * (linspace(trigger_distance, 0, linear_points) / trigger_distance);
        outputs(i,:) = [b_guess, b_linear];
    end
end

%% 保存数据集
save('guidance_dataset_linear.mat', 'inputs', 'outputs', 'trigger_distance');

%% 误差计算函数（恒定b段）
function error = compute_error_constant_b(b, N, v, r0, q0, theta0, theta_f, trigger_dist)
    function dydt = dynamics(~, y)
        r = y(1);
        q = y(2);
        theta = y(3);
        eta = theta - q;
        
        % 恒定b值（即使r<trigger_dist也保持恒定，仅用于第一阶段优化）
        current_b = b;
        
        q_dot = -v * sin(eta) / r;
        theta_dot = N * q_dot + current_b;
        r_dot = -v * cos(eta);
        dydt = [r_dot; q_dot; theta_dot];
    end

    % 提前在800m处停止（因为只关心恒定段的性能）
    function [value, isterminal, direction] = stop_event(~, y)
        value = y(1) - trigger_dist;
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0];
    options = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [~, y] = ode45(@dynamics, [0 1000], y0, options);
    
    if isempty(y)
        error = Inf;
    else
        % 计算800m处的角度与理论值的偏差
        error = (y(end,3) - theta0) - (theta_f - theta0)*(r0 - trigger_dist)/r0;
    end
end

%% 误差计算函数（含线性下降段）
function error = compute_error_variable_b(b, N, v, r0, q0, theta0, theta_f, trigger_dist)
    % 斜率定义
    slope = b / trigger_dist;
    
    function dydt = dynamics(~, y)
        r = y(1);
        q = y(2);
        theta = y(3);
        eta = theta - q;
        
        % 分段b值
        if r >= trigger_dist
            current_b = b;          % 恒定段
        else
            current_b = b * (r / trigger_dist); % 线性下降段
        end
        
        q_dot = -v * sin(eta) / r;
        theta_dot = N * q_dot + current_b;
        r_dot = -v * cos(eta);
        dydt = [r_dot; q_dot; theta_dot];
    end

    % 终止条件（r < 1m）
    function [value, isterminal, direction] = stop_event(~, y)
        value = y(1) - 1;
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0];
    options = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [~, y] = ode45(@dynamics, [0 1000], y0, options);
    
    if isempty(y)
        error = Inf;
    else
        error = y(end,3) - theta_f; % 终端角度误差
    end
end