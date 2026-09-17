import torch
from fontTools.ufoLib import DEFAULT_LAYER_NAME
from pandas.io import feather_format

import main

tensor1 = torch.randint(0,10,(5,5))
tensor2 = torch.randint(0,10,(2,5,5))
# print(tensor1)
print(tensor2)
def f01():

    #返回第一行所有元素
    print(tensor1[1,:])
    print(tensor1[1])

    #返回第二列所有元素
    print(tensor1[ :,2])

def f02():
    #返回对应位置的元素
    print(tensor1[[0,1],[1,2]])

    #广播机制获得0，1行，1，2列四个位置的元素
    print(tensor1[[[0],[1]],[1,2]])

def f03():
    #返回前三行前两列的数据
    print(tensor1[:3,:2])

    #切片，不降维
    print(tensor1[:,:1])

#布尔索引
def f04():
    #第三列的数据大于三的元素对应完整行的数据
    print(tensor1[:,3]>3)
    print(tensor1[tensor1[:,3]>3,:])

    #第二行大于5元素的完整列数据
    print(tensor1[2,:]>5)
    print(tensor1[:,tensor1[2,:]>5])

#多维索引
def fo5():
    #获取第0维（轴）上的第二个数据
    print(tensor2[1,:,:])

    #获取第1维（轴）上的第3个数据
    print(tensor2[:,2,:])

    #获取第2维的第1个数据
    print(tensor2[:,:,0])
if __name__ == '__main__':
    # f01()
    # f02()
    # f03()
    # f04()
    fo5()