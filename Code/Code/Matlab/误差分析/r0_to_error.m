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
%%
% ======================================================================
% 传统解析偏置比例导引律的误差分析代码
% 参考论文：《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）
% 功能：分析初始弹目距离对偏置项计算误差的影响，验证传统解析方法的局限性
% ======================================================================

% ------------------------- 参数设置 -------------------------
r0 = linspace(5000, 10000, 20);  % 初始距离范围：5000~10000 m
qr_deg = -55;                    % 固定视线角变化量（中值条件）
qr = deg2rad(qr_deg);            % 转换为弧度
theta0_deg = -70;                % 初始弹道倾角（度）
theta0 = deg2rad(theta0_deg);    % 转换为弧度
N = 3;                           % 导航系数
v = 300;                         % 导弹速度（m/s）
q0 = 0;                          % 初始视线角（目标在正东方向）

% ------------------------- 预分配存储 -------------------------
b_theory = zeros(size(r0));      % 理论偏置项
b_actual = zeros(size(r0));      % 实际偏置项
absolute_error = zeros(size(r0));% 绝对误差Δb
relative_error = zeros(size(r0));% 相对误差Δb/|b_theory|
terminal_angle_error = zeros(size(r0)); % 终端角度误差（使用理论b时的误差）

% ------------------------- 主循环 -------------------------
for i = 1:length(r0)
    % 期望终端角度θf = q0 + qr（弧度）
    theta_f = q0 + qr;
    
    % 理论飞行时间（公式14）
    tf_theory = r0(i) / v;
    
    % 理论b计算（公式13）
    numerator = N*q0 - theta0 - (N-1)*theta_f;
    b_theory(i) = numerator / tf_theory;
    
    % 通过优化找到实际b使得终端角度误差最小
    options = optimset('TolX', 1e-6, 'Display', 'off');
    b_actual(i) = fzero(@(b) compute_theta_error(b, N, v, r0(i), q0, theta0, theta_f), b_theory(i), options);
    
    % 计算终端角度误差（使用理论b时的误差）
    terminal_angle_error(i) = compute_theta_error(b_theory(i), N, v, r0(i), q0, theta0, theta_f);
    
    % 计算绝对误差和相对误差
    absolute_error(i) = b_actual(i) - b_theory(i);
    relative_error(i) = absolute_error(i) / abs(b_theory(i));
end

% ------------------------- 绘图 -------------------------
% 图1：双纵坐标误差图
figure('Color', 'white', 'Position', [100, 100, 800, 600]);
yyaxis left;
plot(r0, abs(relative_error), 'b-', 'LineWidth', 2);
ylabel('相对误差 \Delta b / |b_{theory}|');
title('初始距离 r_0 对偏置项误差的影响');
grid on;

yyaxis right;
plot(r0, abs(absolute_error), 'r-', 'LineWidth', 2);
ylabel('绝对误差 \Delta b (rad/s)');
legend({'相对误差', '绝对误差'}, 'Location', 'northwest');
xlabel('初始距离 r_0 (m)');

% 图2：终端角度误差图
figure('Color', 'white', 'Position', [100, 100, 800, 600]);
plot(r0, rad2deg(abs(terminal_angle_error)), 'k-', 'LineWidth', 2);
xlabel('初始距离 r_0 (m)');
ylabel('终端角度误差 (°)');
title('理论偏置项 b_{theory} 对应的终端角度误差');
grid on;

% ------------------------- 导弹运动仿真函数 -------------------------
function error = compute_theta_error(b, N, v, r0, q0, theta0, theta_f)
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
        value = y(1) - 1;
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0];
    tspan = [0, 1000];
    options = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [~, y] = ode45(@dynamics, tspan, y0, options);
    
    if isempty(y)
        error = Inf;
    else
        theta_real = y(end, 3);
        error = theta_real - theta_f;
    end
end