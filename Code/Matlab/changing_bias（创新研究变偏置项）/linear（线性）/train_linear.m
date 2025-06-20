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
%% 基于指数衰减模型的变偏置量导引律仿真神经网络训练模块（模型训练模块）
% 创新点：在传统常值偏置基础上，引入距离触发的线性衰减机制
% 参考论文思路：利用非线性映射优化偏置项，提升大范围机动精度

clear; close all; clc;

% load('guidance_dataset.mat'); % 加载inputs和outputs
load('guidance_dataset_linear.mat');

% 数据预处理
[input_rows, input_cols] = size(inputs);

% 数据标准化 (Z-score)
input_mean_linear = mean(inputs, 1);  % 修改为带 _linear
input_std_linear = std(inputs, 0, 1);  % 修改为带 _linear
output_mean_linear = mean(outputs);     % 修改为带 _linear
output_std_linear = std(outputs);       % 修改为带 _linear

normalized_inputs = (inputs - input_mean_linear) ./ input_std_linear;
normalized_outputs = (outputs - output_mean_linear) ./ output_std_linear;

% 划分训练集和+ (8:2)
rng(42); % 固定随机种子
indices = randperm(input_rows);
train_ratio = 0.8;
train_size = round(train_ratio * input_rows);

train_input = normalized_inputs(indices(1:train_size), :);
train_output = normalized_outputs(indices(1:train_size));
val_input = normalized_inputs(indices(train_size+1:end), :);
val_output = normalized_outputs(indices(train_size+1:end));

% 3. 神经网络参数设置
input_size = input_cols;   % 输入特征数 [r0, N, θ0, qr]
hidden_size = 32;          % 隐藏层神经元数（根据论文实验调整）
output_size = 1;           % 输出b值
learning_rate = 0.001;     % 学习率
epochs = 1500;             % 训练轮次
batch_size = 32;           % 批大小

% 4. 初始化网络参数（He初始化）
W1_linear = randn(input_size, hidden_size) * sqrt(2/input_size); % 修改为带 _linear
b1_linear = zeros(1, hidden_size);  % 修改为带 _linear
W2_linear = randn(hidden_size, output_size) * sqrt(2/hidden_size); % 修改为带 _linear
b2_linear = zeros(1, output_size);  % 修改为带 _linear

% 5. Adam优化器参数
beta1 = 0.9;
beta2 = 0.999;  %两个超参数
epsilon = 1e-8; %v的偏置项，很小
m_W1 = zeros(size(W1_linear)); 
v_W1 = zeros(size(W1_linear));
m_b1 = zeros(size(b1_linear)); 
v_b1 = zeros(size(b1_linear));
m_W2 = zeros(size(W2_linear)); 
v_W2 = zeros(size(W2_linear));
m_b2 = zeros(size(b2_linear)); 
v_b2 = zeros(size(b2_linear));

% 6. 训练循环
num_batches = ceil(train_size / batch_size);
train_loss_history = zeros(epochs, 1);
val_loss_history = zeros(epochs, 1);

for epoch = 1:epochs
    % 打乱训练数据
    shuffle_idx = randperm(train_size);
    X = train_input(shuffle_idx, :);
    Y = train_output(shuffle_idx);
    
    batch_loss = 0;
    
    for batch = 1:num_batches
        % 获取当前批次数据
        start_idx = (batch-1)*batch_size + 1;
        end_idx = min(batch*batch_size, train_size);
        X_batch = X(start_idx:end_idx, :);
        Y_batch = Y(start_idx:end_idx);
        
        % 前向传播
        hidden_input = X_batch * W1_linear + b1_linear; % 修改为带 _linear
        hidden_output = tanh(hidden_input); % 使用tanh激活函数
        output = hidden_output * W2_linear + b2_linear; % 修改为带 _linear
        
        % 计算损失（MSE）
        loss = mean((output - Y_batch).^2);
        batch_loss = batch_loss + loss;
        
        % 反向传播
        d_output = 2*(output - Y_batch)/batch_size;
        d_W2 = hidden_output' * d_output;
        d_b2 = sum(d_output, 1);
        
        d_hidden = (d_output * W2_linear') .* (1 - hidden_output.^2); % tanh导数
        d_W1 = X_batch' * d_hidden;
        d_b1 = sum(d_hidden, 1);
        
        % Adam更新
        [W1_linear, b1_linear, W2_linear, b2_linear, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2] = ...
            adam_update(W1_linear, b1_linear, W2_linear, b2_linear, d_W1, d_b1, d_W2, d_b2,...
                        m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2,...
                        learning_rate, beta1, beta2, epsilon, epoch);
    end
    
    % 记录训练损失
    train_loss_history(epoch) = batch_loss / num_batches;
    
    % 验证损失
    val_pred = predict(val_input, W1_linear, b1_linear, W2_linear, b2_linear); % 使用带 _linear
    val_loss = mean((val_pred - val_output).^2);
    val_loss_history(epoch) = val_loss;
    
    % 打印训练进度
    if mod(epoch, 50) == 0
        fprintf('Epoch %d - Train Loss: %.4f | Val Loss: %.4f\n',...
                epoch, train_loss_history(epoch), val_loss);
    end
end

% 7. 保存
save('guidance_model_linear.mat', 'W1_linear', 'b1_linear', 'W2_linear', 'b2_linear', ...
     'input_mean_linear', 'input_std_linear', 'output_mean_linear', 'output_std_linear');

%% 验证代码
% 1. 加载模型和测试数据
load('guidance_model_linear.mat');  % 加载时保持一致
load('guidance_dataset_linear.mat');

% 2. 数据预处理
test_inputs = (inputs - input_mean_linear) ./ input_std_linear; % 使用带 _linear

% 3. 预测
pred_normalized = predict(test_inputs, W1_linear, b1_linear, W2_linear, b2_linear); % 使用带 _linear
predictions = pred_normalized * output_std_linear + output_mean_linear; % 使用带 _linear

% 4. 计算性能指标
mae = mean(abs(predictions - outputs));
rmse = sqrt(mean((predictions - outputs).^2));
fprintf('测试集性能:\nMAE: %.4f\nRMSE: %.4f\n', mae, rmse);

% 5. 可视化结果
figure;
subplot(2,1,1);
plot(1:epochs, train_loss_history, 'b', 1:epochs, val_loss_history, 'r');
title('训练曲线');
xlabel('Epoch');
ylabel('MSE Loss');
legend('训练损失', '验证损失');

subplot(2,1,2);
scatter(outputs, predictions);
hold on;
plot([min(outputs), max(outputs)], [min(outputs), max(outputs)], 'r--');
xlabel('真实值');
ylabel('预测值');
title('预测结果对比');
grid on;

%% 辅助函数
function [W1, b1, W2, b2, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2] = ...
    adam_update(W1, b1, W2, b2, d_W1, d_b1, d_W2, d_b2,...
                m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2,...
                lr, beta1, beta2, epsilon, t)
    % Adam优化器更新参数
    % 权重更新
    [m_W1, v_W1, W1] = update_params(m_W1, v_W1, W1, d_W1, lr, beta1, beta2, epsilon, t);
    [m_b1, v_b1, b1] = update_params(m_b1, v_b1, b1, d_b1, lr, beta1, beta2, epsilon, t);
    [m_W2, v_W2, W2] = update_params(m_W2, v_W2, W2, d_W2, lr, beta1, beta2, epsilon, t);
    [m_b2, v_b2, b2] = update_params(m_b2, v_b2, b2, d_b2, lr, beta1, beta2, epsilon, t);
end

function [m, v, param] = update_params(m, v, param, grad, lr, beta1, beta2, epsilon, t)
    % 单个参数更新
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad.^2;
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);
    param = param - lr * m_hat ./ (sqrt(v_hat) + epsilon);
end

function pred = predict(X, W1, b1, W2, b2)
    % 前向传播预测
    hidden_input = X * W1 + b1;
    hidden_output = tanh(hidden_input);
    pred = hidden_output * W2 + b2;
end
