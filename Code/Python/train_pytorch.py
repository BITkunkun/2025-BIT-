"""
train_pytorch.py - 训练BP神经网络模型用于自适应偏置比例导引

本文件实现论文中描述的BP神经网络训练过程，用于学习参数到偏置项b的映射关系。
使用PyTorch框架构建多层神经网络，采用Adam优化器进行训练。

主要功能：
1. 加载并预处理数据集
2. 定义神经网络结构
3. 训练网络并监控性能
4. 保存训练好的模型
5. 可视化训练结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 加载数据集
data = torch.load('guidance_dataset_python_3w.pt')
inputs = data['inputs']    # 输入特征 [r0, N, θ0, qr]
outputs = data['outputs']  # 输出目标 [b]

# 2. 数据预处理
input_rows, input_cols = inputs.shape

# 数据标准化(Z-score归一化)
input_mean = torch.mean(inputs, dim=0)
input_std = torch.std(inputs, dim=0)
output_mean = torch.mean(outputs)
output_std = torch.std(outputs)

normalized_inputs = (inputs - input_mean) / input_std
normalized_outputs = (outputs - output_mean) / output_std

# 划分训练集和验证集(8:2比例)
torch.manual_seed(42)  # 固定随机种子保证可重复性
indices = torch.randperm(input_rows)
train_size = int(0.8 * input_rows)

train_input = normalized_inputs[indices[:train_size]]
train_output = normalized_outputs[indices[:train_size]]
val_input = normalized_inputs[indices[train_size:]]
val_output = normalized_outputs[indices[train_size:]]

# 转换为PyTorch Dataset和DataLoader
batch_size = 512  # 批处理大小
train_dataset = TensorDataset(train_input, train_output)
val_dataset = TensorDataset(val_input, val_output)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 3. 定义神经网络结构(3隐藏层)
class MultiLayerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化多层神经网络
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层神经元数量
            output_size: 输出维度
        """
        super(MultiLayerNetwork, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(hidden_size, hidden_size) # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(hidden_size, hidden_size) # 隐藏层2到隐藏层3
        self.fc4 = nn.Linear(hidden_size, output_size) # 隐藏层3到输出层
        self.sigmoid = nn.Sigmoid()  # 激活函数
        
        # Xavier初始化权重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        """前向传播"""
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.fc4(x)  # 输出层不使用激活函数
        return x

# 4. 初始化模型和优化器
input_size = input_cols    # 输入特征数 [r0, N, θ0, qr]
hidden_size = 15           # 每层15个神经元(论文中设置)
output_size = 1            # 输出b值
learning_rate = 0.001      # 学习率(Adam优化器)

# 设置设备(GPU优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiLayerNetwork(input_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 5. 训练循环
epochs = 5000  # 训练轮数
train_loss_history = []  # 记录训练损失
val_loss_history = []    # 记录验证损失

for epoch in range(epochs):
    model.train()  # 训练模式
    batch_losses = []
    
    # 批量训练
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()  # 梯度清零
        predictions = model(X_batch)  # 前向传播
        loss = criterion(predictions, Y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        batch_losses.append(loss.item())
    
    # 计算平均训练损失
    train_loss = np.mean(batch_losses)
    train_loss_history.append(train_loss)
    
    # 验证损失(整批验证)
    model.eval()  # 评估模式
    with torch.no_grad():
        val_input, val_output = val_input.to(device), val_output.to(device)
        val_pred = model(val_input)
        val_loss = criterion(val_pred, val_output).item()
        val_loss_history.append(val_loss)
    
    # 打印训练进度(每100轮)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

# 6. 保存模型(包含标准化参数)
torch.save({
    'model_state_dict': model.state_dict(),  # 模型参数
    'input_mean': input_mean,  # 输入均值(用于反标准化)
    'input_std': input_std,    # 输入标准差
    'output_mean': output_mean, # 输出均值
    'output_std': output_std   # 输出标准差
}, 'guidance_model_multiLayers_pytorch_3w.pth')

# 7. 可视化训练结果
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_loss_history, 'b', label='训练损失')
plt.plot(range(epochs), val_loss_history, 'r', label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('训练曲线')
plt.legend()

# 预测结果可视化
model.eval()
with torch.no_grad():
    predictions = model(val_input)
    # 反标准化
    predictions = predictions.cpu() * output_std + output_mean
    targets = val_output.cpu() * output_std + output_mean

plt.subplot(1, 2, 2)
plt.scatter(targets.numpy(), predictions.numpy())
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测结果对比')
plt.tight_layout()
plt.show()

# 8. 计算性能指标
mae = torch.mean(torch.abs(predictions - targets))  # 平均绝对误差
rmse = torch.sqrt(torch.mean((predictions - targets)**2))  # 均方根误差
print(f'测试集性能:\nMAE: {mae.item():.6f}\nRMSE: {rmse.item():.6f}')
