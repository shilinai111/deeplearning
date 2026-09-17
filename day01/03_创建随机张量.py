import torch
import matplotlib.pyplot as plt
#查看当前随机数种子
# print(torch.initial_seed())

#查看当前随机数种子
torch.manual_seed(13)
print(torch.initial_seed())
def f01():
    out = torch.rand(300,500)
    print(out)
    #画分布图
    out = out.flatten()
    # print(out.shape)
    plt.xlim(-0.1,1.2)  #横坐标范围
    plt.hist(out)   #柱状图
    plt.show()

#演示标准高斯分布的随即张量的生成
def f02():
    out = torch.randn(300,500)
    #画分布图
    out = out.flatten()
    plt.xlim(-3,3)  #横坐标范围
    plt.hist(out,bins = 100)   #柱状图,bins增加柱状图的数字数
    plt.show()

#创建指定范围的随机整数张量
def f03():
    out = torch.randint(0,100,(300,500))
    print(out,out.shape)
    # 画分布图  1 对数据进行展平操作  2D 变成1D
    out = out.flatten()
    # print(out.shape)

    plt.xlim(-10, 120)  # 将图像的横坐标显示范围缩小到 -5~25
    plt.hist(out)  # 对15万元素的1D张量 画柱状图
    plt.show()


if __name__ == '__main__':
    # f01()
    # # # f02()
    f03()