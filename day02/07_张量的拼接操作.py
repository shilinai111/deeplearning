import torch


#除拼接维度外，其他维度行数必须相同
#拼接操作不会升维
def f01():
    data1 = torch.randint(0,10,(2,3,4))
    data2 = torch.randint(0,10,(5,3,4))
    data3 = torch.randint(0,10,(2,3,4))
    #开始拼接，指定拼接维度
    #参1 指定凭借的多个张量，参2：指定拼接的维度

    out = torch.cat((data1,data3),dim=1)
    print(out.shape)

#演示stack 堆积
#两个堆叠的张量形状完全一样，堆叠后会升维

def f02():
    data1 = torch.randint(0,10,(2,3,4))
    data2 = torch.randint(0,10,(2,3,4))
    #参1 指定堆叠的多个张量，参2：需要指定新增的维度是哪一维
    out = torch.stack((data1,data2),dim=0)
    print(out.shape)

    out1 = torch.stack((data1,data2),dim=1)
    print(out1.shape)

    out2 = torch.stack((data1,data2),dim=2)
    print(out2.shape)

if __name__ == '__main__':
    f01()
    f02()