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

%% 基于指数衰减模型的变偏置量导引律仿真验证模块“打击静止目标”
% 创新点：在传统常值偏置基础上，引入距离触发的指数衰减机制

% 统一参数设置
N = 3;                          % 导航系数
v = 300;                        % 导弹速度 (m/s)
r0 = 10000;                     % 初始弹目距离 (m)
theta0_deg = 10;                % 初始弹道倾角 (°)
q0 = 0;                         % 初始视线角 (rad)，目标在正前方
x0 = 0;
h0 = 0;

qr_deg_values = [-40, -60, -80]; % 期望终端交会角
num_qr = length(qr_deg_values);

% 初始化存储变量
results_cbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'a_M_end', {}, 'tt', {}, 'theta_values', {});
results_nncbpng = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'a_M_end', {}, 'tt', {}, 'theta_values', {});
results_adaptive = struct('b', {}, 'terminal_angle', {}, 'end_distance', {}, 'a_M_end', {}, 'tt', {}, 'theta_values', {});

% 变b参数
trigger_distance = 800;  % b开始衰减的距离 (m)
k = 10;                   % 指数衰减系数

for i = 1:num_qr
    qr_deg = qr_deg_values(i);
    load('guidance_model.mat', 'input_mean', 'input_std', 'output_mean', 'output_std', 'W1', 'b1', 'W2', 'b2');
    %% 1. 传统CBPNG理论b值（公式13）
    theta0 = deg2rad(theta0_deg);
    qr = deg2rad(qr_deg);
    theta_f = q0 + qr;
    t_go = r0 / v;
    b_theory = (N*q0 - theta0 - (N-1)*theta_f) / t_go;

    %% 2. 神经网络定b值
    input_nn = [r0, N, theta0_deg, qr_deg];
    input_normalized = (input_nn - input_mean) ./ input_std;
    b_pred_nn = predict(input_normalized, W1, b1, W2, b2) * output_std + output_mean;

    %% 3. 变b模型预测初始b值（使用exp模型参数）
    load('guidance_model_exp.mat', 'input_mean', 'input_std', 'output_mean', 'output_std', 'W1', 'b1', 'W2', 'b2');
    input_nn_exp = [r0, N, theta0_deg, qr_deg];
    input_normalized_exp = (input_nn_exp - input_mean) ./ input_std;
    b_pred_exp = predict(input_normalized_exp, W1, b1, W2, b2) * output_std + output_mean;

    %% 仿真三种情况（严格区分逻辑）
    % 情况1：传统定b（固定b_theory）
    [results_cbpng(i).terminal_angle, results_cbpng(i).end_distance, results_cbpng(i).tt,...
     results_cbpng(i).theta_values, results_cbpng(i).n_end] = simulate_fixed_b(b_theory, N, v, r0, q0, theta0);
    
    % 情况2：神经网络定b（固定b_pred_nn）
    [results_nncbpng(i).terminal_angle, results_nncbpng(i).end_distance, results_nncbpng(i).tt,...
     results_nncbpng(i).theta_values, results_nncbpng(i).n_end] = simulate_fixed_b(b_pred_nn, N, v, r0, q0, theta0);
    
    % 情况3：变b模型（动态调整b_pred_exp）
    [results_adaptive(i).terminal_angle, results_adaptive(i).end_distance, results_adaptive(i).tt,...
     results_adaptive(i).theta_values, results_adaptive(i).n_end] = simulate_adaptive_b(b_pred_exp, N, v, r0, q0, theta0, trigger_distance, k);
end

%% 结果可视化
figure('Color','white','Position',[100 100 1000 800]);

% ========== 弹道倾角对比 ==========
hold on;
colors = lines(3); 
linestyles = {'--','-',':'}; 
markers = {'o','s','^'};
labels = {'CBPNG (定b)','NNCBPNG (定b)','Adaptive-b (变b)'};

% 绘制所有曲线（9条）
for i = 1:num_qr
    % CBPNG
    plot(results_cbpng(i).tt, rad2deg(results_cbpng(i).theta_values),...
        'Color',colors(1,:), 'LineStyle',linestyles{1}, 'LineWidth',1,...
        'Marker',markers{1}, 'MarkerIndices',1:100:length(results_cbpng(i).tt),...
        'DisplayName',[labels{1} ' qr=' num2str(qr_deg_values(i)) '°']);
    
    % NNCBPNG
    plot(results_nncbpng(i).tt, rad2deg(results_nncbpng(i).theta_values),...
        'Color',colors(2,:), 'LineStyle',linestyles{2}, 'LineWidth',1,...
        'Marker',markers{2}, 'MarkerIndices',1:100:length(results_nncbpng(i).tt),...
        'DisplayName',[labels{2} ' qr=' num2str(qr_deg_values(i)) '°']);
    
    % Adaptive
    plot(results_adaptive(i).tt, rad2deg(results_adaptive(i).theta_values),...
        'Color',colors(3,:), 'LineStyle',linestyles{3}, 'LineWidth',1,...
        'Marker',markers{3}, 'MarkerIndices',1:100:length(results_adaptive(i).tt),...
        'DisplayName',[labels{3} ' qr=' num2str(qr_deg_values(i)) '°']);
end

title('弹道倾角对比（九曲线）');
xlabel('时间 (s)');
ylabel('弹道倾角θ (°)');
legend('show','Location','eastoutside');
grid on;
hold off;

% 调整图例（避免重复）
hLeg = findobj(gcf,'Type','Legend');
set(hLeg(1),'Position',[0.85 0.7 0.1 0.2]); % 调整弹道倾角图例位置

fprintf('\n==================== 命中点法向过载对比 ====================\n');
for i = 1:num_qr
    fprintf('CBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_cbpng(i).n_end);
    fprintf('NNCBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_nncbpng(i).n_end);
    fprintf('NNVBPNG (qr = %.0f°) 命中点法向过载：%.4f\n', qr_deg_values(i), results_adaptive(i).n_end);
end
%% ==================== 严格分离的仿真函数 ====================
% 定b模型仿真函数
function [terminal_angle, miss_distance, tt, theta_values, n_end] = simulate_fixed_b(b, N, v, r0, q0, theta0)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        eta = theta - q;
        
        % 定b逻辑：b全程不变
        q_dot = -v * sin(eta) / r;
        theta_dot = N * q_dot + b;  
        a_M = v * theta_dot;  % 法向加速度
        
        r_dot = -v * cos(eta);
        x_dot = v*cos(theta);
        h_dot = v*sin(theta);
        
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot; a_M];
    end

    function [value, isterminal, direction] = stop_event(~, y)
        value = y(5);
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0; 0; 0; 0];
    tspan = [0, 1000];
    opts = odeset('Events',@stop_event, 'RelTol',1e-6);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    
    terminal_angle = y(end,3);
    theta_values = y(:,3);

    theta_dot_end = abs((y(end, 3)-y((end-1),3))/(tt(end)-tt(end-1))); 
    % 注意此处不能直接调用最后一个theta_dot，需要用这种方式近似处理，
    
    %theta_dot_end = abs(N * q_dot_end + b1);
    n_end = v*theta_dot_end/9.8;
    miss_distance = sqrt((y(end,4)-10000)^2 + y(end,5)^2); 
end

% 变b模型仿真函数
function [terminal_angle, miss_distance, tt, theta_values, n_end] = simulate_adaptive_b(b_initial, N, v, r0, q0, theta0, trigger_distance, k)
    function dydt = dynamics(~, y)
        r = y(1);    q = y(2);    theta = y(3);   
        eta = theta - q;
        
        % 动态调整
        k = 10;
        trigger_distance=800;
        if r <= trigger_distance
            decay_factor = exp(-k*(trigger_distance - r)/trigger_distance);
            b_current = b_initial * decay_factor;
        else
            b_current = b_initial;
        end
        
        q_dot = -v * sin(eta) / r;
        theta_dot = N * q_dot + b_current;  
        a_M = v * theta_dot;  % 法向加速度
        
        r_dot = -v * cos(eta);
        x_dot = v*cos(theta);
        h_dot = v*sin(theta);
        
        dydt = [r_dot; q_dot; theta_dot; x_dot; h_dot; a_M];
    end

    function [value, isterminal, direction] = stop_event(~, y)
        value = y(5);
        isterminal = 1;
        direction = -1;
    end

    y0 = [r0; q0; theta0; 0; 0; 0];
    tspan = [0, 1000];
    opts = odeset('Events',@stop_event, 'RelTol',1e-6);
    [tt, y] = ode45(@dynamics, tspan, y0, opts);
    
    terminal_angle = y(end,3);
    theta_values = y(:,3);

    theta_dot_end = abs((y(end, 3)-y((end-1),3))/(tt(end)-tt(end-1))); 
    % 注意此处不能直接调用最后一个theta_dot，需要用这种方式近似处理，
    
    %theta_dot_end = abs(N * q_dot_end + b1);
    n_end = v*theta_dot_end/9.8;

    miss_distance = sqrt((y(end,4)-10000)^2 + y(end,5)^2); 
end

%% 神经网络预测函数（保持不变）
function pred = predict(X, W1, b1, W2, b2)
    hidden_input = X * W1 + b1;
    hidden_output = tanh(hidden_input);
    pred = hidden_output * W2 + b2;
end