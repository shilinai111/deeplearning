

import torch
import torch.nn as nn
from sympy.vector import orienters


def f01():

    #定义真实值和预测值
    y_ture = torch.tensor([2.0,2.0,2.0],dtype=torch.float)
    y_pred = torch.tensor([1.0,1.2,1.9],dtype=torch.float)


    orienters = nn.L1Loss()

    loss = orienters(y_pred,y_ture)
    print(f'L1Loss:{loss}')


def f02():
    # 定义真实值和预测值
    y_ture = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.2, 1.9], dtype=torch.float)

    orienters = nn.MSELoss()

    loss = orienters(y_pred, y_ture)
    print(f'MSELoss:{loss}')


def f03():
    # 定义真实值和预测值
    y_ture = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.2, 1.9], dtype=torch.float)

    orienters = nn.SmoothL1Loss()

    loss = orienters(y_pred, y_ture)
    print(f'SmoothL1Loss:{loss}')

if __name__ == '__main__':
    f01()
    f02()
    f03()