import torch
import numpy as np


def f01():
    tem = torch.randint(0,10,(3,5))
    # out = tem.numpy()#内存共享 浅拷贝
    out = tem.numpy().copy()#内存独立，深拷贝

    print(out,out.shape,type(out))

    tem[0,0] = -99
    print(out)

def f02():
    tem = np.array([1,2,3,4,5,6])
    print(tem,tem.shape,type(tem))

    out = torch.from_numpy(tem).clone()
    tem[0] = -99
    print(out, out.shape, type(out))

def f03():
    tem = torch.rand(1)
    print(tem)
    a = tem.item()
    print(a,type(a))

if __name__ == '__main__':
    # f01()
    # f02()
    f03()
