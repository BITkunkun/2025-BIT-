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

%% 基于线性衰减模型的变偏置量导引律仿真验证模块“打击静止目标”（验证模型模块）
% 创新点：在传统常值偏置基础上，引入距离触发的线性衰减机制

% 统一参数设置（导弹与目标参数）
N = 3;                          % 导航系数
v = 300;                        % 导弹速度 (m/s)
r0 = 10000;                     % 初始弹目距离 (m)
theta0_deg = 10;                % 初始弹道倾角 (°)
q0 = 0;                         % 初始视线角 (rad)，目标在正前方
x0 = 0;                         % 导弹初始x位置
h0 = 0;                         % 导弹初始高度

% 目标运动参数（新增）
VT = 0;                        % 目标速度 (m/s) 
thetaT = 0;                     % 目标运动方向 (rad)，0表示水平向右

qr_deg_values = [-40, -60, -80]; % 期望终端交会角
num_qr = length(qr_deg_values);

% 初始化存储变量
results_cbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'end_ny', {}, 'tt', {}, 'theta_values', {});
results_nncbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'end_ny', {}, 'tt', {}, 'theta_values', {});
results_nnvbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'end_ny', {}, 'tt', {}, 'theta_values', {}); % 新增

% 加载神经网络模型和标准化参数（注意输入需包含目标运动参数）
load('guidance_model.mat');     % 确保模型训练时输入为[r0, N, theta0_deg, qr_deg, VT, thetaT_deg]
load('guidance_model_linear.mat'); % 加载新模型

for i = 1:num_qr
    qr_deg = qr_deg_values(i);
    
    % 传统CBPNG理论b值（公式需考虑目标运动）
    theta0 = deg2rad(theta0_deg);
    qr = deg2rad(qr_deg);
    theta_f = q0 + qr;          % 终端视线角=终端弹道倾角 (rad)
    t_go = r0 / (v + VT*cos(thetaT)); % 修正预估飞行时间
    b_theory = (N*q0 - theta0 - (N-1)*theta_f) / t_go;

    % 神经网络b值（输入需包含目标参数）
   input_nn = [r0, N, theta0_deg, qr_deg];  % 保持4个输入
   input_normalized = (input_nn - input_mean) ./ input_std;
   b_pred_nn = predict(input_normalized, W1, b1, W2, b2) * output_std + output_mean;

    % NNVBPNG神经网络新模型的 b 值预测
    input_nn_linear = [r0, N, theta0_deg, qr_deg];  
    input_normalized_linear = (input_nn_linear - input_mean_linear) ./ input_std_linear;    
    b_pred_nn_linear = predict(input_normalized_linear, W1_linear, b1_linear, W2_linear, b2_linear) * output_std_linear + output_mean_linear;  

    % 仿真
    [results_cbpng(i).terminal_angle, results_cbpng(i).end_distance, results_cbpng(i).tt, results_cbpng(i).theta_values, results_cbpng(i).end_ny] = ...
        simulate_missile_flight(b_theory, N, v, r0, q0, theta0, x0, h0, VT, thetaT);
    
    [results_nncbpng(i).terminal_angle, results_nncbpng(i).end_distance, results_nncbpng(i).tt, results_nncbpng(i).theta_values, results_nncbpng(i).end_ny] = ...
        simulate_missile_flight(b_pred_nn, N, v, r0, q0, theta0, x0, h0, VT, thetaT);
    
    [results_nnvbpng(i).terminal_angle, results_nnvbpng(i).end_distance, results_nnvbpng(i).tt, results_nnvbpng(i).theta_values, results_nnvbpng(i).end_ny] = ...
        simulate_missile_flight_linear(b_pred_nn_linear, N, v, r0, q0, theta0, x0, h0, VT, thetaT);
end

%% 结果输出与绘图
fprintf('==================== 终端落角对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), rad2deg(results_cbpng(i).terminal_angle));
    fprintf('NNCBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), rad2deg(results_nncbpng(i).terminal_angle));
    fprintf('NNVBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), rad2deg(results_nnvbpng(i).terminal_angle)); % 新增
end

fprintf('\n==================== 脱靶量对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_cbpng(i).end_distance);
    fprintf('NNCBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_nncbpng(i).end_distance);
    fprintf('NNVBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_nnvbpng(i).end_distance); % 新增
end
fprintf('\n==================== 命中点法向过载对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_cbpng(i).end_ny);
    fprintf('NNCBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_nncbpng(i).end_ny);
    fprintf('NNVBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_nnvbpng(i).end_ny);
end
%% 绘制θ变化曲线
figure('Color', 'white', 'Position', [100 100 800 600]);
hold on;
% 扩展颜色数组以适用于多条曲线
colors = { ...
    [0.2, 0.4, 0.8], ...  % 颜色1
    [0.6, 0.8, 1.0], ...
    [0.2, 0.6, 0.8], ...  % 颜色2
    [0.8, 0.2, 0.2], ...  % 颜色3
    [1.0, 0.6, 0.6], ...  % 颜色4
    [0.3, 0.4, 0.8], ...
    [0.1, 0.6, 0.1], ...  % 颜色5
    [0.6, 0.9, 0.6], ...  % 颜色6
    [0.8, 0.6, 0.2], ...  % 颜色8（新颜色）
    [0.9, 0.3, 0.5], ...  % 颜色9（新颜色）
}; 
linestyles = {'--', '-',':'};
labels = {'CBPNG', 'NNCBPNG', 'NNVBPNG'}; % 更新标签
line_width = 1;

curve_index = 1;
for i = 1:num_qr
    % 绘制CBPNG曲线
    linestyle_index = 1;
    plot(results_cbpng(i).tt, rad2deg(results_cbpng(i).theta_values),...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index}, 'LineWidth', line_width,...
         'DisplayName', [labels{1} ', qr = ' num2str(qr_deg_values(i)) '°']);
    curve_index = curve_index + 1;

    % 绘制NNCBPNG曲线
    linestyle_index = 2; 
    plot(results_nncbpng(i).tt, rad2deg(results_nncbpng(i).theta_values),...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index}, 'LineWidth', line_width,...
         'DisplayName', [labels{2} ', qr = ' num2str(qr_deg_values(i)) '°']);
    curve_index = curve_index + 1;

    % 绘制NNVBPNG曲线
    linestyle_index = 3; 
    plot(results_nnvbpng(i).tt, rad2deg(results_nnvbpng(i).theta_values),...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index}, 'LineWidth', line_width,...
         'DisplayName', [labels{3} ', qr = ' num2str(qr_deg_values(i)) '°']); % 新增
    curve_index = curve_index + 1;
end

xlabel('时间 (s)');
ylabel('弹道倾角 θ (°)');
title('打击静止目标弹道倾角θ变化曲线');
legend('show', 'Location', 'best');
grid on;
hold off;

%% 动力学仿真函数（关键修改：加入目标运动）
function [terminal_angle, miss_distance, tt, theta_values, end_ny] = simulate_missile_flight(b, N, v, r0, q0, theta0, x0, h0, VT, thetaT)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        x_missile = y(4);  h_missile = y(5);
        
        % 相对运动方程（考虑目标运动）
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
        value = y(5); % 导弹高度h=0时终止
        isterminal = 1;
        direction = -1;
    end
    
    y0 = [r0; q0; theta0;x0;h0];
    tspan = [0, 1000];
    opts = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    terminal_angle = y(end, 3);
    theta_values = y(:, 3);

    eta_end = y(end, 3) - y(end, 2);
    q_dot_end = (-VT*sin(y(end, 2)-thetaT) - v*sin(eta_end)) / y(end, 1);
    theta_dot_end = abs((y(end, 3)-y((end-1),3))/(tt(end)-tt(end-1)));
    %theta_dot_end = abs(N * q_dot_end + b);
    end_ny = v*theta_dot_end/9.8;
    miss_distance = y(end, 4)-10000-tt(end)*VT; %脱靶量计算
end

function [terminal_angle, miss_distance, tt, theta_values, end_ny] = simulate_missile_flight_linear(b, N, v, r0, q0, theta0, x0, h0, VT, thetaT)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        x_missile = y(4);  h_missile = y(5);
        trigger_distance = 800;
        % 相对运动方程（考虑目标运动）
        eta = theta - q;
        q_dot = (-VT*sin(q-thetaT) -v * sin(eta)) / r;
        % 记录当前距离和b值
        if r <= trigger_distance
            b1 = b*r/trigger_distance;
            theta_dot = N * q_dot + b1;
        else
            theta_dot = N * q_dot + b;
        end
        r_dot = VT*cos(q-thetaT) - v*cos(eta);        % 修正距离变化率
        
        % 导弹绝对运动
        x_dot = v*cos(theta);
        h_dot = v*sin(theta);
        
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot];
    end
    
    function [value, isterminal, direction] = stop_event(~, y)
        value = y(5); % 导弹高度h=0时终止
        isterminal = 1;
        direction = -1;
    end
    
    y0 = [r0; q0; theta0;x0;h0];
    tspan = [0, 1000];
    opts = odeset('Events', @stop_event, 'RelTol', 1e-6);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    terminal_angle = y(end, 3);
    theta_values = y(:, 3);
    
    b1 = b*y(end, 1)/800;

    theta_dot_end = abs((y(end, 3)-y((end-1),3))/(tt(end)-tt(end-1)));
    %theta_dot_end = abs(N * q_dot_end + b1);
    end_ny = v*theta_dot_end/9.8;

    miss_distance = y(end, 4)-10000-tt(end)*VT; %脱靶量计算
end

% 3. 神经网络预测函数
function pred = predict(X, W1, b1, W2, b2)
    hidden_input = X * W1 + b1;
    hidden_output = tanh(hidden_input);
    pred = hidden_output * W2 + b2;
end 