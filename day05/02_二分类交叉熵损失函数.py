
import torch
import torch.nn as nn

#创建标签 三个样本
y_true = torch.tensor([0.,1.,0],dtype = torch.float)

#预测值，需要得到sigmoid函数进行处理后的  归一化分类概率（不能输入分类得分）
y_pred = torch.tensor([0.3,0.5,0.78],dtype = torch.float)


criterion = nn.BCELoss()#不含softmax需要转化为概率

loss = criterion(y_pred,y_true)#要取平均值，防止样本过多梯度爆炸
print(f'loss:{loss}')