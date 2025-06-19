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
% 对应论文第4.3节考虑导引头不确定性的"高斯白噪声"

% 参数设置
N = 3;
v = 300;
r0 = 10000;
theta0_deg = 10;
q0 = 0;
x0 = 0;
h0 = 0;

qr_deg_values = [-40, -60, -80];
num_qr = length(qr_deg_values);

% 初始化存储变量
results_cbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});
results_nncbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});
results_cbpng_noise = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});
results_nncbpng_noise = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'tt', {}, 'theta_values', {});

% 加载神经网络模型和标准化参数
load('guidance_model.mat');

% 高斯白噪声参数
noise_mean = 0;
noise_std = 0.6 * pi / 180;

for i = 1:num_qr
    qr_deg = qr_deg_values(i);
    
    % 传统CBPNG理论b值
    theta0 = deg2rad(theta0_deg);
    qr = deg2rad(qr_deg);
    theta_f = q0 + qr;
    t_go = r0 / v;
    b_theory = (N*q0 - theta0 - (N-1)*theta_f) / t_go;

    % 神经网络b值
    input_nn = [r0, N, theta0_deg, qr_deg];
    input_normalized = (input_nn - input_mean) ./ input_std;
    b_pred_nn = predict(input_normalized, W1, b1, W2, b2) * output_std + output_mean;

    % 情况1：理论b值仿真（无噪声，传入默认噪声参数）
    [results_cbpng(i).terminal_angle, results_cbpng(i).end_distance, results_cbpng(i).tt, results_cbpng(i).theta_values] = simulate_missile_flight(b_theory, N, v, r0, q0, theta0, x0, h0, 0, 0);
    % 情况2：神经网络b值仿真（无噪声，传入默认噪声参数）
    [results_nncbpng(i).terminal_angle, results_nncbpng(i).end_distance, results_nncbpng(i).tt, results_nncbpng(i).theta_values] = simulate_missile_flight(b_pred_nn, N, v, r0, q0, theta0, x0, h0, 0, 0);

    % 情况1：理论b值仿真（有噪声）
    [results_cbpng_noise(i).terminal_angle, results_cbpng_noise(i).end_distance, results_cbpng_noise(i).tt, results_cbpng_noise(i).theta_values] = simulate_missile_flight(b_theory, N, v, r0, q0, theta0, x0, h0, noise_mean, noise_std);
    % 情况2：神经网络b值仿真（有噪声）
    [results_nncbpng_noise(i).terminal_angle, results_nncbpng_noise(i).end_distance, results_nncbpng_noise(i).tt, results_nncbpng_noise(i).theta_values] = simulate_missile_flight(b_pred_nn, N, v, r0, q0, theta0, x0, h0, noise_mean, noise_std);
end

%% 结果整理与输出
% 转换为度数
for i = 1:num_qr
    results_cbpng(i).terminal_angle_deg = rad2deg(results_cbpng(i).terminal_angle);
    results_nncbpng(i).terminal_angle_deg = rad2deg(results_nncbpng(i).terminal_angle);
    results_cbpng_noise(i).terminal_angle_deg = rad2deg(results_cbpng_noise(i).terminal_angle);
    results_nncbpng_noise(i).terminal_angle_deg = rad2deg(results_nncbpng_noise(i).terminal_angle);
end

% 打印终端落角和脱靶量
fprintf('==================== 终端落角对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), results_cbpng(i).terminal_angle_deg);
    fprintf('NNCBPNG (qr = %.0f°) 落角：%.4f°\n', qr_deg_values(i), results_nncbpng(i).terminal_angle_deg);
    fprintf('CBPNG (qr = %.0f°，有噪声) 落角：%.4f°\n', qr_deg_values(i), results_cbpng_noise(i).terminal_angle_deg);
    fprintf('NNCBPNG (qr = %.0f°，有噪声) 落角：%.4f°\n', qr_deg_values(i), results_nncbpng_noise(i).terminal_angle_deg);
end

fprintf('\n==================== 脱靶量对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_cbpng(i).end_distance);
    fprintf('NNCBPNG (qr = %.0f°) 脱靶量：%.6f m\n', qr_deg_values(i), results_nncbpng(i).end_distance);
    fprintf('CBPNG (qr = %.0f°，有噪声) 脱靶量：%.6f m\n', qr_deg_values(i), results_cbpng_noise(i).end_distance);
    fprintf('NNCBPNG (qr = %.0f°，有噪声) 脱靶量：%.6f m\n', qr_deg_values(i), results_nncbpng_noise(i).end_distance);
end

%% 绘制高斯白噪声值的图
noise_values = noise_mean + noise_std * randn(1000, 1);
figure('Color', 'white', 'Position', [100 100 800 600]);
plot(1:1000, rad2deg(noise_values));
xlabel('样本点');
ylabel('高斯白噪声值 (°)');
title('导引头测量信息中的高斯白噪声值');
grid on;

%% 绘制θ变化曲线（加入噪声后）
figure('Color', 'white', 'Position', [100 100 800 600]);
hold on;
colors = { [0.2, 0.4, 0.8], [0.6, 0.8, 1.0], [0.8, 0.2, 0.2], [1.0, 0.6, 0.6], [0.1, 0.6, 0.1], [0.6, 0.9, 0.6]}; 
linestyles = {'--', '-'};
labels = {'CBPNG', 'NNCBPNG'};
line_width = 1;

curve_index = 1;
for i = 1:num_qr
    % 绘制CBPNG曲线（有噪声）
    linestyle_index = mod(curve_index - 1, 2) + 1;
    plot(results_cbpng_noise(i).tt, results_cbpng_noise(i).terminal_angle_deg,...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index}, 'LineWidth', line_width,...
         'DisplayName', [labels{1} ', qr = ' num2str(qr_deg_values(i)) '°，有噪声']);
    curve_index = curve_index + 1;

    % 绘制NNCBPNG曲线（有噪声）
    linestyle_index = mod(curve_index - 1, 2) + 1;
    plot(results_nncbpng_noise(i).tt, results_nncbpng_noise(i).terminal_angle_deg,...
         'Color', colors{curve_index}, 'LineStyle', linestyles{linestyle_index}, 'LineWidth', line_width,...
         'DisplayName', [labels{2} ', qr = ' num2str(qr_deg_values(i)) '°，有噪声']);
    curve_index = curve_index + 1;
end

xlabel('时间 (s)');
ylabel('弹道倾角 θ (°)');
title('加入高斯白噪声后弹道倾角θ变化曲线');
legend('show', 'Location', 'best');
grid on;
hold off;

% 动力学仿真函数
function [terminal_angle, miss_distance, tt, theta_values] = simulate_missile_flight(b, N, v, r0, q0, theta0, x0, h0, noise_mean, noise_std)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        % x = y(4);  h = y(5)  ;
        eta = theta - q;
        % 加入高斯白噪声
        noise = noise_mean + noise_std * randn;
        q_dot = (-v * sin(eta) / r) + noise;
        theta_dot = N * q_dot + b;
        r_dot = -v * cos(eta);
        x_dot = v*cos(theta);
        h_dot = v*sin(theta);
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot];
    end
    function [value, isterminal, direction] = stop_event(~, y)
        value = y(5);
        isterminal = 1;
        direction = -1;
    end
    y0 = [r0; q0; theta0; x0; h0];
    tspan = [0, 1000];
    opts = odeset('Events', @stop_event, 'RelTol', 1e-4);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    terminal_angle = y(end, 3);
    theta_values = y(:, 3);
    
    miss_distance = sqrt((y(end,4) - 10000)^2 + y(end,5)^2);% 脱靶量
end

% 3. 神经网络预测函数
function pred = predict(X, W1, b1, W2, b2)
    hidden_input = X * W1 + b1;
    hidden_output = tanh(hidden_input);
    pred = hidden_output * W2 + b2;
end