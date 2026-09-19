import torch
import torch.nn as nn
from torch import softmax

#创建标签  两个样本
# y_true = torch.tensor([[0,1,0],[0,0,1]],dtype=torch.float)
y_true = torch.tensor([1,2])

#分类网络的输出结果（各个类别上的 分类得分 logit）
y_pred = torch.tensor([[-0.02,2.5,0.2],[0.6,0.8,-1.2]])

#softmax结果
soft = nn.Softmax(dim=1)
y_pred1 = soft(y_pred)
print(y_pred1)


#损失函数实例化
criterion = nn.CrossEntropyLoss()  #内涵softmax

#计算损失值
loss = criterion(y_pred,y_true)
print(f'loss:{loss}')