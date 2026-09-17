import torch
import matplotlib.pyplot as plt


#低温
tao = 0.2

#高温
tao = 5
scores = torch.tensor([0.2,0.02,0.15,0.15,1.3,0.5,0.06,1.1,0.05,3.75])/tao

#计算归一化概率
prob = torch.softmax(scores,dim=0)

print(prob)