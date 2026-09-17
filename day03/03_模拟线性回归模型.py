
import torch
from torch.utils.data import TensorDataset  #构造数据集对象
from torch.utils.data import DataLoader     #数据加载器
from torch import nn                        #nn模块又平方损失函数和假设函数
from torch import optim                     #optim模块又优化器函数
from sklearn.datasets import make_regression    #船舰线性回归数据集模型
import matplotlib.pyplot as plt
import numpy as np

def create_dataset():
    x,y,coef = make_regression(
        n_samples=100,#100个样本数据点
        n_features=1,#输入的特征数只有1
        noise=10,    #高斯分布噪音的标准差。越大分部越离散
        coef=True,   #是否返回模型参数w
        bias=14.5,  #固定截距为14.5 可调
        random_state=16,
    )
    # print(x,x.shape)
    # print(y,y.shape)
    # print(coef)

    #讲numpy array类型的数据变成张量才能用于训练
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x , y , coef

def plot_img(x,y,coef,bias):
    #绘制散点图
    plt.scatter(x,y)

    #获得两个x坐标点（最大值最小值）
    X = [x.min().item(),x.max().item()]

    #利用极值点的横坐标，计算纵坐标
    y1 = [i*coef + bias for i in X]

    plt.plot(X,y1,label = 'real',color = 'r')
    plt.grid()
    plt.legend()
    plt.show()

def train(x, y):
    #第一个2
    #指定数据集对象和数据加载器
    dataset = TensorDataset(x,y)

    # 数据加载器
    # 参1  数据集对象(告诉加载器，单个样本如何加载和预处理，从而使之可以批量生成)
            #告诉加载器: 训练数据总量是多少
    # 参2  批次大小(作用1生成对应批次的数据张量，作用2让加载器自己计算出，一轮需要迭代多少个批次)
    #
    # 参3  shuffe: 每一轮开始前，打乱数据，避免模型从样本送入的顺序中学到脏的特征(只有训练集   需要开启)
    # 参4   是否丢弃最后的不满批次的数据(训练集如果稀少正规，则不丢弃)
    dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True,drop_last=True)
    #第一个4 定义四个组件（模型，损失函数，优化器，学习率调度器）
    #nn.Linear()是神经网络的线性层，或者叫全连接层
    #参数1 in_feature
    #参数2 out_feature
    model = nn.Linear(in_features=1,out_features=1)

    #损失函数
    criterion = nn.MSELoss()

    #定义优化器
    #参数1 ：指定模型那些参数被优化器管理
    #参数2 ：初始学习率
    optimizer = optim.SGD(model.parameters(),lr=2e-3)

    #原则上需要定义 学习率衰减策略（调度器）

    #完成第二个2 两层循环
    epochs = 300 #训练轮数
    #创建一个空列表 用于存放每一轮的（各个批次平均后的）损失值
    epoch_loss = []

    #外层对轮次的循环
    for i in range(epochs):

        total = 0

        #内层对每一轮中个个批次的数据进行循环
        for train_x , train_y in dataloader:
            # print(train_x,train_y,train_x.dtype)
            # print(train_y,train_y.shape)

            #正向传播  特征x的元素类型需要从double转为float(32bit)才能正向传播
            y_pred = model(train_x.float())

            #将真实值的形状，让其变成2d张量，从而与预测值形状匹配，得以计算损失
            train_y = train_y.reshape(-1,1).float()

            #计算loss值   (为了1   评估模型好坏，  2  为了计算梯度)
            loss = criterion(y_pred, train_y)# 一个批次16个样本的loss的平均值
            # print(loss)

            #第二个4
            optimizer.zero_grad() #使用优化器，对所有参数的梯度清0
            loss.backward()#反向传播
            optimizer.step()#每个参数根据梯度，进行更新

            total += loss.item()

        #学习率调度器判断，完成学习率衰减
        #lr_scheduler.step()
        
        #计算当前轮训练完后，所有批次loss的平均值
        epoch_loss.append(total/len(dataloader))
        
        #每一轮训练完，获得模型更新完的参数
        #derach()作用：将张量从计算图中剥离，不计算梯度
        weight_pred = model.weight.detach().item()
        bias_pred = model.bias.detach().item()
        print(i,weight_pred,bias_pred)

    plt.plot(range(epochs),epoch_loss)
    plt.savefig('epoch_loss.png')
    plt.show()
    return weight_pred, bias_pred


def plot_img2(x, y, coef, bias,weight_pred,bias_pred):
    # 绘制散点图
    plt.scatter(x, y)

    # 获得两个x坐标点（最大值最小值）
    X = [x.min().item(), x.max().item()]

    # 利用极值点的横坐标，计算纵坐标--理论上的
    y1 = [i * coef + bias for i in X]

    #训练出来的
    y2 = [i * weight_pred + bias_pred for i in X]

    #理论模型
    plt.plot(X, y1, label='real', color='r')

    #训练模型
    plt.plot(X, y2, label='trained', color='b')

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x,y,coef = create_dataset()
    # plot_img2(x,y,coef,14.5)
    weight_pred, bias_pred = train(x, y)
    plot_img2(x, y, coef, 14.5, weight_pred, bias_pred)