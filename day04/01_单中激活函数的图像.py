# Tanh:将输入映射成-1~1之间的值
#       梯度的最大值为1 因此依然在链式求导中会产生梯度消失现象。
#       RNN,LSTM,GRU中使用
# ReLU: 热门的分段线性激活函数，解决了梯度消失和梯度爆炸现象，
#       但是会带来神经元死亡现象(详细说一下， 稀疏性， 但严重死亡)
#       因此被后续改进为LeakyReLU  PReLU等
#

import torch
import matplotlib.pyplot as plt

x = torch.linspace(-20,20,1000)

#将自变量下送入激活函数，获得激活值
y = torch.sigmoid(x)
y = torch.tanh(x)

# plt.plot(x,y)
# plt.show()

#绘制激活函数的梯度图像
#需要计算自变量x的梯度，因此在定义线性张量的时，候需要打开梯度

x = torch.linspace(-20,20,1000,requires_grad=True)

#进行正向传播
y = torch.sigmoid(x)
print(y.shape)
#反向传播 y先变为标量张量，否则无法开启梯度
y.sum().backward()
print(y.sum().shape)

# plt.plot(x.data,x.grad)
# plt.show()