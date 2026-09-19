import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置超参
element_number = 30
beta = 0.9


# 创建平均气温数据
def f01():
    # 固定种子
    torch.manual_seed(19)

    # 从标准的高斯分布中抽取随机数，产生30天的随机气温
    temperature = torch.randn(element_number) * 10
    print(temperature)

    # 绘图
    days = torch.arange(1, element_number + 1, 1)
    plt.plot(days, temperature, color='red')
    plt.scatter(days, temperature)
    plt.show()


def f02():
    torch.manual_seed(19)
    temperature = torch.randn(element_number) * 10
    # 创建一个空列表，用于 记录每个时刻的指数加权移动平均值
    ema = []
    # 通过迭代，计算每个时刻的指数加权移动平均值 EMA
    for i, temp in enumerate(temperature, 1):  # enumerate 同时获得 索引以及 内容
        # print(i,temp)  #展示每天的天数和 气温
        if i == 1:
            ema.append(temp)
            continue
        new_temp = beta * ema[-1] + (1 - beta) * temp  # 根据公式获得当前时刻的 EMA值
        ema.append(new_temp)


# 绘制图像
    days = torch.arange(1, element_number + 1, 1)
    plt.plot(days, ema, color='r')
    plt.scatter(days, temperature)
    plt.show()

if __name__ == '__main__':
    # f01()
    f02()
