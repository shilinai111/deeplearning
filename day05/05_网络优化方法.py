import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

#演示动量优化器
def f01():

    #初始化模型的权重参数，需要计算梯度，因此需要打开
    w = torch.tensor([1.0],requires_grad=True)

    #定义损失函数
    loss = w**2/2.0 #梯度=w

    #定义优化器
    #参1 指定需要优化的参数范围 参2 初始学习率 参3 动量发的beta超参（这个值越大，月参考历史价值，波动越小）
    optimizer = optim.SGD([w],lr = 0.01,momentum = 0.9)

    #清空梯度
    optimizer.zero_grad()
    #反向传播，计算梯度
    loss.backward()
    #参数更新
    optimizer.step()

    #打印
    print(w.detach().numpy(),w.grad.numpy())

#第二个批次
    # 定义损失函数
    loss = w ** 2 / 2.0  # 梯度=w

    # 清空梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 参数更新
    optimizer.step()

    #打印
    print(w.detach().numpy(),w.grad.numpy())

if __name__ == '__main__':
    f01()
