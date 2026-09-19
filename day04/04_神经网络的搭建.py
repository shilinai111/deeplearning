import torch
import torch.nn as nn
from sympy.logic import inference
from torchsummary import summary

#根据要求搭建神经网络

class ModelDemo(nn.Module):
    def __init__(self):
        #初始化父类
        super().__init__()

        #定义神经网络第一层
        self.linear1 = nn.Linear(in_features=3, out_features=3)

        #第二层隐藏层
        self.linear2 = nn.Linear(in_features=3, out_features=2)

        #定义神经网络第三层输出层
        self.output = nn.Linear(in_features=2, out_features=2)

        #对隐藏层各层进行初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        out = torch.softmax(self.output(x), dim=-1)
        return out

def inference1():
    #实例化模型，创建模型对象，调用__init__()方法，完成初始化
    my_model = ModelDemo()

    #创建数据张量
    data = torch.randn((5,3))
    print(data,data.shape)

    #正向传播
    pred = my_model(data)

    print(pred,pred.shape)

    #分析神经网络的结构，有那些层，每层有多少参数，层与层之间的顺序
    #参1  那个模型需要分析    参2  指定输入张量的形状  参3  指定在那个设备进行分析
    summary(my_model, input_size=(5,3), device='cpu')

if __name__ == '__main__':
    inference1()