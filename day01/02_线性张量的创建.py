import torch

def f01():

    #参1:等差数列的起始值，参2:等差数列的末尾值,参3:步长，参4:数据类型
    #arange包头不包尾
    out = torch.arange(0,12,3,dtype=torch.float32)
    print(out,out.shape,out.dtype)

def f02():
    # 参1:等差数列的起始值，参2:等差数列的末尾值,参3:结果张量中元素的总数，参4:数据类型
    out = torch.linspace(0,12,3,dtype=torch.float32)
    print(out, out.shape, out.dtype)

if __name__ == '__main__':
    f01()
    f02()
