# Guidance Model Research Project

## 项目简介
本项目包含Matlab和Python实现的制导模型研究代码，主要涉及：
- 基础论文复现
- 误差分析
- 创新研究（变偏置项）
- 灵敏度分析

## 研究背景
（请在此处补充研究背景和目的）

## 项目结构

### Matlab目录
```
Matlab/
├── 误差分析/               # 误差分析相关代码
│   ├── LOS_to_error.m
│   └── r0_to_error.m
├── basic_基础论文复现/      # 基础论文复现代码
│   ├── Guass.m
│   ├── guidance_dataset.mat
│   ├── guidance_model*.mat
│   ├── MonteCarlo_simulation*.m
│   ├── movingX.m
│   ├── movingY.m
│   ├── preparing.m
│   ├── train_multiLayers_paper.m
│   └── validate.m
├── changing_bias（创新研究变偏置项）/  # 创新研究
│   ├── exp（指数）/         # 指数变化研究
│   └── linear（线性）/      # 线性变化研究
└── theta0灵敏度分析/        # 灵敏度分析
    ├── qr灵敏度分析/
    └── r0灵敏度分析/
    └── theta0灵敏度分析/
```

### Python目录
```
Python/
├── Guass_noise.py          # 高斯噪声处理
├── guidance_dataset_python_3w.pt
├── guidance_model_multiLayers_pytorch_3w.pth
├── Monte_carlo_simulation_pytorch.py  # 蒙特卡洛仿真
├── preparing.py            # 数据准备
├── requirements.txt        # Python依赖
├── train_pytorch.py        # 训练脚本
└── validate.py             # 验证脚本
```

## 安装和运行要求

### Matlab要求
- MATLAB R2020a或更高版本
- 需要安装的工具箱: (请补充)

### Python要求
```bash
pip install -r requirements.txt
```

## 各模块功能说明

1. **基础论文复现**
   - `MonteCarlo_simulation_MultiLayers_paper.m`: 多层模型的蒙特卡洛仿真
   - `train_multiLayers_paper.m`: 多层模型训练脚本

2. **创新研究（变偏置项）**
   - 包含指数和线性两种变化方式的研究代码

3. **灵敏度分析**
   - 对qr、r0和theta0参数的灵敏度分析

4. **误差分析**
   - LOS和r0误差分析

## 贡献指南
欢迎通过Pull Request贡献代码。请确保:
1. 代码风格与现有代码一致
2. 添加适当的注释
3. 更新相关文档

## 许可证
（请在此处补充许可证信息，如MIT License）
