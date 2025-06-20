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

%% 目标运动场景下的制导律对比仿真模块
% 参考论文：《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）
% 功能：在目标垂直运动场景下对比传统CBPNG与NNCBPNG的制导性能
% 扩展内容：新增目标运动速度VT、运动方向thetaT，修正相对运动方程（目标y方向运动）
% 对应论文第4章4.2节仿真验证部分“击低匀速目标时与 CBPNG 对比验证”

%% 统一参数设置
N = 3;                          
v = 300;                        
r0 = 10000;                    
theta0_deg = 10;              
q0 = 0;                     
x0 = 0;
h0 = 0;
VT = 10;                      
thetaT = deg2rad(90);           % 目标运动方向，90°表示垂直向上
qr_deg_values = [-40, -60, -80];
num_qr = length(qr_deg_values);

results_cbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});
results_nncbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});

% 加载神经网络模型和标准化参数
load('guidance_model.mat');

for i = 1:num_qr
    qr_deg = qr_deg_values(i);
    
    % 传统CBPNG理论b值（考虑垂直运动）
    theta0 = deg2rad(theta0_deg);   % 初始弹道倾角 (rad)
    qr = deg2rad(qr_deg);           % 视线角变化量 (rad)
    theta_f = q0 + qr;              % 终端视线角=终端弹道倾角 (rad)
    t_go = r0 / (v + VT*cos(thetaT-q0)); % 修正预估飞行时间
    b_theory = (N*q0 - theta0 - (N-1)*theta_f) / t_go;

    % 神经网络b值
    input_nn = [r0, N, theta0_deg, qr_deg];
    input_normalized = (input_nn - input_mean) ./ input_std;
    b_pred_nn = predict(input_normalized, W1, b1, W2, b2) * output_std + output_mean;

    % 情况1：理论b值仿真
    [results_cbpng(i).terminal_angle, results_cbpng(i).end_distance, results_cbpng(i).tt, results_cbpng(i).t_end,...
     results_cbpng(i).theta_values, results_cbpng(i).x_traj, results_cbpng(i).h_traj] = ...
        simulate_missile_flight(b_theory, N, v, r0, q0, theta0, x0, h0, VT, thetaT);
    
    % 情况2：神经网络b值仿真
    [results_nncbpng(i).terminal_angle, results_nncbpng(i).end_distance, results_nncbpng(i).tt, results_nncbpng(i).t_end,...
     results_nncbpng(i).theta_values, results_nncbpng(i).x_traj, results_nncbpng(i).h_traj] = ...
        simulate_missile_flight(b_pred_nn, N, v, r0, q0, theta0, x0, h0, VT, thetaT);

end

%% 结果整理与输出
% 转换为度数
for i = 1:num_qr
    results_cbpng(i).terminal_angle_deg = rad2deg(results_cbpng(i).terminal_angle);
    results_nncbpng(i).terminal_angle_deg = rad2deg(results_nncbpng(i).terminal_angle);
end

% 打印终端落角和脱靶量
fprintf('==================== 终端落角对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), results_cbpng(i).terminal_angle_deg);
    fprintf('NNCBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), results_nncbpng(i).terminal_angle_deg);
end

fprintf('\n==================== 脱靶量对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_cbpng(i).end_distance);
    fprintf('NNCBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_nncbpng(i).end_distance);
end

%% 绘制θ变化曲线
figure('Color', 'white', 'Position', [100 100 800 600]);
hold on;
colors = { [0.2, 0.4, 0.8], [0.6, 0.8, 1.0], [0.8, 0.2, 0.2], [1.0, 0.6, 0.6], [0.1, 0.6, 0.1], [0.6, 0.9, 0.6]}; 
linestyles = {'--', '-'};
labels = {'CBPNG', 'NNCBPNG'};
line_width = 1;

curve_index = 1;
for i = 1:num_qr
    % 绘制CBPNG曲线
    linestyle_index = mod(curve_index - 1, 2) + 1;
    plot(results_cbpng(i).tt, rad2deg(results_cbpng(i).theta_values),...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index},...
         'LineWidth', line_width,...
         'DisplayName', [labels{1} ', qr = ' num2str(qr_deg_values(i)) '°']);
    curve_index = curve_index + 1;

    % 绘制NNCBPNG曲线
    linestyle_index = mod(curve_index - 1, 2) + 1;
    plot(results_nncbpng(i).tt, rad2deg(results_nncbpng(i).theta_values),...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index},...
         'LineWidth', line_width,...
         'DisplayName', [labels{2} ', qr = ' num2str(qr_deg_values(i)) '°']);
    curve_index = curve_index + 1;
end

xlabel('时间 (s)');
ylabel('弹道倾角 θ (°)');
title('垂直运动目标下弹道倾角θ变化曲线');
legend('show', 'Location', 'best');
grid on;
hold off;

%% 绘制导弹与目标轨迹
figure('Color', 'white', 'Position', [100 100 1200 800]);
hold on;

% 颜色和线型定义
colors = { [1, 0, 0], [0, 0, 1], [1, 0.5, 0], [0, 0.5, 1], [0.5, 0, 0.5], [0, 0.7, 0] }; % 红、蓝、橙、浅蓝、紫、绿
linestyles = {'--', '-'};
labels = {'CBPNG', 'NNCBPNG'};

% 绘制导弹轨迹
for i = 1:num_qr
    % CBPNG轨迹
    plot(results_cbpng(i).x_traj, results_cbpng(i).h_traj, ...
        'Color', colors{i}, 'LineStyle', linestyles{1}, 'LineWidth', 1.5, ...
        'DisplayName', [labels{1} ', qr = ' num2str(qr_deg_values(i)) '°']);
    
    % NNCBPNG轨迹
    plot(results_nncbpng(i).x_traj, results_nncbpng(i).h_traj, ...
        'Color', colors{i+num_qr}, 'LineStyle', linestyles{2}, 'LineWidth', 1.5, ...
        'DisplayName', [labels{2} ', qr = ' num2str(qr_deg_values(i)) '°']);
end

% 绘制目标轨迹（目标水平运动）
t_sim_max = max(max([results_cbpng.t_end; results_nncbpng.t_end])); % 获取最大仿真时间
t_target = 0:1:t_sim_max;
x_target = 10000 + VT * t_target * cos(thetaT);
y_target = 0 + VT * t_target * sin(thetaT);
plot(x_target, y_target, 'k-', 'LineWidth', 2, 'DisplayName', '目标轨迹');

% 图形标注
xlabel('水平位置 (m)');
ylabel('高度 (m)');
title('导弹与目标轨迹对比');
legend('show', 'Location', 'best');
grid on;
axis equal;
hold off;

%% 动力学仿真函数（考虑目标运动）
function [terminal_angle, miss_distance, tt, t_end, theta_values, x_traj, h_traj] = simulate_missile_flight(b, N, v, r0, q0, theta0, x0, h0, VT, thetaT)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        % x_missile = y(4);  h_missile = y(5);
        
        % 相对运动方程
        eta = theta - q;
        q_dot = (-VT*sin(q-thetaT) - v*sin(eta)) / r;  % 修正视线角速率
        theta_dot = N * q_dot + b;
        r_dot = VT*cos(q-thetaT) - v*cos(eta);        % 修正距离变化率
        
        % 导弹绝对运动
        x_dot = v*cos(theta);
        h_dot = v*sin(theta);
        
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot];
    end
    
    function [value, isterminal, direction] = stop_event(~, y)
        value = y(1);
        % value = y(4) - 10000;
        isterminal = 1;
        direction = -1;
    end
    
    y0 = [r0; q0; theta0; x0; h0];
    tspan = [0, 1000];
    opts = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    
    terminal_angle = y(end, 3);
    theta_values = y(:, 3);
    x_traj = y(:, 4);   % 导弹x轨迹
    h_traj = y(:, 5);   % 导弹y轨迹
    
    t_end = tt(end);
    miss_distance = y(end, 5)-tt(end)*VT; %脱靶量计算
end

%% 神经网络预测函数
function pred = predict(X, W1, b1, W2, b2)
    hidden_input = X * W1 + b1;
    hidden_output = tanh(hidden_input);
    pred = hidden_output * W2 + b2;
end