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

%% 基于BP神经网络的自适应偏置比例导引蒙特卡洛仿真
% 参考论文：《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）
% 功能：通过蒙特卡洛方法评估NNCBPNG在不同初始条件下的鲁棒性
% 对应论文第4.3节"蒙特卡洛仿真验证"

% 蒙特卡洛仿真参数
num_samples = 500;                % 样本数量
v = 300;                          % 导弹速度 (m/s)
N = 3;                            % 导航系数
missile_pos = [0, 0];             % 导弹固定初始位置
target_x_range = [5000, 10000];   % 目标初始x坐标范围 (m)
theta0_range = [10, 20];          % 初始发射角范围 (°)
theta_f_range = [-90, -50];       % 期望终端交会角范围 (°)

% 加载神经网络模型和标准化参数
load('guidance_model_multiLayers_paper.mat');

errors = zeros(num_samples, 1);   % 终端交会角误差

% 生成随机样本参数
target_x_values = target_x_range(1)+(target_x_range(2)-target_x_range(1))*rand(num_samples,1);
theta0_values = theta0_range(1)+(theta0_range(2)-theta0_range(1))*rand(num_samples,1);
theta_f_desired_values = theta_f_range(1)+(theta_f_range(2)-theta_f_range(1))*rand(num_samples,1);

% 蒙特卡洛主循环
for i = 1:num_samples
    % 当前样本参数
    target_pos = [target_x_values(i), 0];
    theta0_deg = theta0_values(i);
    theta_f_desired_deg = theta_f_desired_values(i);

    % NNCBPNG偏置项计算
    input_nn = [norm([missile_pos(1)-target_pos(1),missile_pos(2)-target_pos(2)]), N, theta0_deg, theta_f_desired_deg];
    input_normalized = (input_nn - input_mean)./input_std;
    
    % 修改预测函数调用
    b_pred = predict(input_normalized, W1, b1, W2, b2, W3, b3, W4, b4)*output_std + output_mean;

    % 运动学仿真
    [theta_f_actual_deg, ~]=simulate_missile_flight(b_pred, N, v, norm([missile_pos(1)-target_pos(1),missile_pos(2)-target_pos(2)]), theta0_deg, target_pos);

    % 误差计算
    errors(i)=abs(theta_f_actual_deg - theta_f_desired_deg);
end

max_error = max(errors);
min_error = min(errors);
mean_error = mean(errors);
std_error = std(errors);

% 输出结果
fprintf('=============== 蒙特卡洛仿真结果 ===============\n');
fprintf('最大误差: %.4f°\n', max_error);
fprintf('最小误差: %.4f°\n', min_error);
fprintf('平均误差: %.4f°\n', mean_error);
fprintf('标准误差: %.4f°\n\n', std_error);

% 绘制误差散点图
figure('Color','white');
scatter(1:num_samples, errors, 'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0.2,0.4,0.8]);
xlabel('次数');
ylabel('终端交会角误差 (°)');
title('NNCBPNG终端交会角误差分布');
grid on;

% 动力学仿真函数
function [theta_f_actual_deg, miss_distance]=simulate_missile_flight(b, N, v, r0, theta0_deg, target_pos)
    % 初始条件
    theta0_rad = deg2rad(theta0_deg);
    q0 = 0;                % 初始视线角
    x0 = 0;
    h0 = 0;

    % 运动学方程
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        eta = theta - q;
        q_dot = -v * sin(eta)/r;
        theta_dot = N * q_dot + b;
        r_dot = -v * cos(eta);
        x_dot = v * cos(theta);
        h_dot = v * sin(theta);
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot];
    end

    % 停止条件（弹目距离<1m）
    function [value, isterminal, direction]=stop_event(~, y)
        value = y(1) - 1;  % r < 1m时停止
        isterminal = 1;
        direction = -1;
    end

    % 数值求解
    y0 = [r0; q0; theta0_rad; x0; h0];
    tspan = [0, 1000];
    opts = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [~, y]=ode45(@dynamics, tspan, y0, opts);

    % 提取结果
    theta_f_actual_deg = rad2deg(y(end,3));
    miss_distance = sqrt((y(end,4)-target_pos(1))^2+(y(end,5)-target_pos(2))^2);
end

% 神经网络预测函数
function pred = predict(X, W1, b1, W2, b2, W3, b3, W4, b4)
    hidden_input1 = X * W1 + b1;
    hidden_output1 = 1./(1 + exp(-hidden_input1));
    hidden_input2 = hidden_output1 * W2 + b2;
    hidden_output2 = 1./(1 + exp(-hidden_input2));
    hidden_input3 = hidden_output2 * W3 + b3;
    hidden_output3 = 1./(1 + exp(-hidden_input3));
    pred = hidden_output3 * W4 + b4;
end