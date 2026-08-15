# 卫星图像变化检测（Siamese-FCN

##项目简介
用孪生全卷积网络做LEVIR-CD数据集的变化检测任务。
输入同一区域不同时的相的两张卫星图,输出变化区域标注。

## 环境依赖
python 3.x ,  PyTorch , torchvision, PIL,matplolib, numy

##数据集
[LEVIR-CD](https://github.com/justchenhao/LEVIR_CD)
- 训练集；645对图像
- 测试集 204对图像
- 图像尺寸 1024x1024 (训练时裁剪到256x256)

## 模型结构
- Encoder: 4层CNN(共享权重的孪生网络)
- Decoder: 4层上采样 + Skip Connection
- 损失函数: BCE + Dice Loss

#训练结果
- epoch:100
- IoU: 0.293
- 准确率: 98.4%

##使用方法
python train.py
