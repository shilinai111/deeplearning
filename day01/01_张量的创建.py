import torch
import numpy as np

#利用torch.tensor吧数据转化为张量

def f01():
    #标量
    out = torch.tensor(5,)
    print(out,type(out),out.shape,out.dtype)

    #列表
    out2 = torch.tensor([5,6,7,8,9,10])
    print(out2,type(out2))

    #数组
    out3 = torch.tensor(np.array([5.,6,7,8,9,10]))
    print(out3,type(out3))

def f02():
    out = torch.Tensor(10,10)
    #张量，形状，数据结构，元素类型
    print(out,out.shape,type(out),out.dtype)

    #初始化
    # out.zero_() #讲未初始化的张量。所有元素填入0，下划线代表变量的原位修改，不用等号
    out.fill_(100)
    print(out)

    #numpy数组
    out3 = torch.Tensor(np.array([5.,6,7,8,9,10]))
    print(out3,type(out3))

# 整型全精度浮点数 双精度浮点数张量的创建
def f03():
    out = torch.IntTensor(10,10).zero_()
    print(out,out.shape,type(out),out.dtype)

    #将长列表转化为整型张量.
    out = torch.IntTensor([1.8,2.4,-5.9])
    print(out,out.shape,type(out),out.dtype)

    #生成单精度浮点数元素的张量 float32 未初始化
    out2 = torch.FloatTensor(20,20)
    print(out2, out2.shape, type(out2), out2.dtype)

    # 生成双精度浮点数元素的张量 float32 未初始化
    out3 = torch.DoubleTensor(20,20)
    print(out3, out3.shape, type(out3), out3.dtype)

#其他元素类型api
def f04():
    data = torch.ShortTensor()#int16 短整型 模型压缩 (量化   蒸馏   剪枝)
    data = torch.IntTensor()#int32
    data = torch.LongTensor()#int64
    data = torch.HalfTensor()#半精度浮点数  float16
    data = torch.FloatTensor()#全精度浮点数  float32
    data = torch.DoubleTensor()#双精度浮点数  float64

if __name__ == '__main__':
    # f01()
    f02()
    # f03()
