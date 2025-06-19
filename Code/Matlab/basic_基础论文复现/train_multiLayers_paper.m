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
%% 基于BP神经网络的自适应偏置比例导引算法训练模块
% 参考论文：《基于BP神经网络的自适应偏置比例导引》（刘畅等，兵工学报，2022）
% 功能：实现BP神经网络的训练、验证与性能评估，对应论文第3章（神经网络模型构建）
% 训练代码对应论文第3章“BP神经网络的训练”

% 数据加载与预处理
clear; close all; clc;
load('guidance_dataset.mat');

% 数据标准化 (Z-score)
[input_rows, input_cols] = size(inputs);
input_mean = mean(inputs, 1);
input_std = std(inputs, 0, 1);
output_mean = mean(outputs);
output_std = std(outputs);

normalized_inputs = (inputs - input_mean) ./ input_std;
normalized_outputs = (outputs - output_mean) ./ output_std;

% 划分训练集和验证集 (8:2)
rng(42); 
indices = randperm(input_rows);
train_ratio = 0.8;
train_size = round(train_ratio * input_rows);

train_input = normalized_inputs(indices(1:train_size), :);
train_output = normalized_outputs(indices(1:train_size));
val_input = normalized_inputs(indices(train_size+1:end), :);
val_output = normalized_outputs(indices(train_size+1:end));

% 神经网络参数设置
input_size = input_cols;   % 输入特征数 [r0, N, θ0, qr]
hidden_size = 15;          % 每层15个神经元（3个隐含层）
output_size = 1;           % 输出b值
learning_rate = 0.001;    
epochs = 5000;            
batch_size = 32;          

% （Xavier初始化）
W1 = randn(input_size, hidden_size) * sqrt(1/input_size); % Xavier
b1 = zeros(1, hidden_size);
W2 = randn(hidden_size, hidden_size) * sqrt(1/hidden_size);
b2 = zeros(1, hidden_size);
W3 = randn(hidden_size, hidden_size) * sqrt(1/hidden_size);
b3 = zeros(1, hidden_size);
W4 = randn(hidden_size, output_size) * sqrt(1/hidden_size);
b4 = zeros(1, output_size);

% Adam优化器参数
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1));
m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1));
m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2));
m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2));
m_W3 = zeros(size(W3)); v_W3 = zeros(size(W3));
m_b3 = zeros(size(b3)); v_b3 = zeros(size(b3));
m_W4 = zeros(size(W4)); v_W4 = zeros(size(W4));
m_b4 = zeros(size(b4)); v_b4 = zeros(size(b4));

% 训练循环
num_batches = ceil(train_size / batch_size);
train_loss_history = zeros(epochs, 1);
val_loss_history = zeros(epochs, 1);

for epoch = 1:epochs
    % 打乱训练数据(避免过拟合)
    shuffle_idx = randperm(train_size);
    X = train_input(shuffle_idx, :);
    Y = train_output(shuffle_idx);
    
    batch_loss = 0;
    
    for batch = 1:num_batches
        start_idx = (batch-1)*batch_size + 1;
        end_idx = min(batch*batch_size, train_size);
        X_batch = X(start_idx:end_idx, :);
        Y_batch = Y(start_idx:end_idx);

        % --- 前向传播（使用sigmoid激活函数）---
        hidden_input1 = X_batch * W1 + b1;
        hidden_output1 = 1./(1 + exp(-hidden_input1));
        
        hidden_input2 = hidden_output1 * W2 + b2;
        hidden_output2 = 1./(1 + exp(-hidden_input2));
        
        hidden_input3 = hidden_output2 * W3 + b3;
        hidden_output3 = 1./(1 + exp(-hidden_input3));
        
        output = hidden_output3 * W4 + b4; 
    
        % --- 反向传播 ---
        d_output = 2*(output - Y_batch)/batch_size;
        
        d_hidden3 = (d_output * W4') .* (hidden_output3 .* (1 - hidden_output3)); % Sigmoid导数
        d_W3 = hidden_output2' * d_hidden3;
        d_b3 = sum(d_hidden3, 1);
        
        d_hidden2 = (d_hidden3 * W3') .* (hidden_output2 .* (1 - hidden_output2));
        d_W2 = hidden_output1' * d_hidden2;
        d_b2 = sum(d_hidden2, 1);
        
        d_hidden1 = (d_hidden2 * W2') .* (hidden_output1 .* (1 - hidden_output1));
        d_W1 = X_batch' * d_hidden1;
        d_b1 = sum(d_hidden1, 1);
        
        d_W4 = hidden_output3' * d_output;
        d_b4 = sum(d_output, 1);

        % Adam更新
        [W1, b1, W2, b2, W3, b3, W4, b4, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2, m_W3, v_W3, m_b3, v_b3, m_W4, v_W4, m_b4, v_b4] = ...
            adam_update(W1, b1, W2, b2, W3, b3, W4, b4, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_b4,...
                        m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2, m_W3, v_W3, m_b3, v_b3, m_W4, v_W4, m_b4, v_b4,...
                        learning_rate, beta1, beta2, epsilon, epoch);
    end
    
    % 记录训练损失（MSE）
    train_loss_history(epoch) = batch_loss / num_batches;
    
    val_pred = predict(val_input, W1, b1, W2, b2, W3, b3, W4, b4);
    val_loss = mean((val_pred - val_output).^2);
    val_loss_history(epoch) = val_loss;
    
    if mod(epoch, 50) == 0
        fprintf('Epoch %d - Train Loss: %.6f | Val Loss: %.6f\n',...
                epoch, train_loss_history(epoch), val_loss);
    end
end

% 保存模型
save('guidance_model_multiLayers_paper.mat', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4',...
     'input_mean', 'input_std', 'output_mean', 'output_std');

%% 验证代码
% 1. 加载模型和测试数据
load('guidance_model_multiLayers_paper.mat');
load('guidance_dataset.mat');

% 2. 数据预处理
test_inputs = (inputs - input_mean) ./ input_std; % 标准化

% 3. 预测
pred_normalized = predict(test_inputs, W1, b1, W2, b2, W3, b3, W4, b4);
predictions = pred_normalized * output_std + output_mean;

% 4. 计算性能指标
mae = mean(abs(predictions - outputs));
rmse = sqrt(mean((predictions - outputs).^2));
fprintf('测试集性能:\nMAE: %.6f\nRMSE: %.6f\n', mae, rmse);

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
function [W1, b1, W2, b2, W3, b3, W4, b4, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2, m_W3, v_W3, m_b3, v_b3, m_W4, v_W4, m_b4, v_b4] = ...
    adam_update(W1, b1, W2, b2, W3, b3, W4, b4, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_W4, d_b4,...
                m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2, m_W3, v_W3, m_b3, v_b3, m_W4, v_W4, m_b4, v_b4,...
                lr, beta1, beta2, epsilon, t)
    % Adam优化器更新参数
    % 权重更新
    [m_W1, v_W1, W1] = update_params(m_W1, v_W1, W1, d_W1, lr, beta1, beta2, epsilon, t);
    [m_b1, v_b1, b1] = update_params(m_b1, v_b1, b1, d_b1, lr, beta1, beta2, epsilon, t);
    [m_W2, v_W2, W2] = update_params(m_W2, v_W2, W2, d_W2, lr, beta1, beta2, epsilon, t);
    [m_b2, v_b2, b2] = update_params(m_b2, v_b2, b2, d_b2, lr, beta1, beta2, epsilon, t);
    [m_W3, v_W3, W3] = update_params(m_W3, v_W3, W3, d_W3, lr, beta1, beta2, epsilon, t);
    [m_b3, v_b3, b3] = update_params(m_b3, v_b3, b3, d_b3, lr, beta1, beta2, epsilon, t);
    [m_W4, v_W4, W4] = update_params(m_W4, v_W4, W4, d_W4, lr, beta1, beta2, epsilon, t);
    [m_b4, v_b4, b4] = update_params(m_b4, v_b4, b4, d_b4, lr, beta1, beta2, epsilon, t);
end

function [m, v, param] = update_params(m, v, param, grad, lr, beta1, beta2, epsilon, t)
    % 单个参数更新
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad.^2;
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);
    param = param - lr * m_hat ./ (sqrt(v_hat) + epsilon);
end

%% 预测函数
function pred = predict(X, W1, b1, W2, b2, W3, b3, W4, b4)
    % 前向传播（使用sigmoid激活函数）
    hidden_input1 = X * W1 + b1;
    hidden_output1 = 1./(1 + exp(-hidden_input1));
    
    hidden_input2 = hidden_output1 * W2 + b2;
    hidden_output2 = 1./(1 + exp(-hidden_input2));
    
    hidden_input3 = hidden_output2 * W3 + b3;
    hidden_output3 = 1./(1 + exp(-hidden_input3));
    
    pred = hidden_output3 * W4 + b4;
end