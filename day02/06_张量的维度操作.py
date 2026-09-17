import torch

#掌握reshape()
def f01():
    data = torch.tensor([[10,20,30],[40,50,60]])

    #打印张量的形状
    print(data.shape,data.shape[0],data.shape[1])
    print(data.size(),data.size(0),data.size(1))

    tensor = torch.rand(2,3,4)
    # out = tensor.reshape(12,2)
    out = tensor.reshape(-1,4)
    print(out.shape)

    #扁平化 flatten操作 将数据进行降维
    x = torch.rand(2,3,4,5,6,7,8)
    out = x.flatten()
    print(out.shape)

    out = torch.flatten(x,start_dim=2,end_dim=4)
    print(out.shape)

#张量的升维和降维  unsqueeze和squeeze
def f02():
    tensora = torch.rand(2,1,3,4,1,5,1,6)
    print(tensora.squeeze().shape)

    #利用unsqueeze对张量进行升维，unsqueeze会对现有的张量新增一个大小为1的维度
    #需要指定邢增的轴再张量中的第几个维度
    tensor1 = torch.rand(2,3,4,5)
    out = tensor1.unsqueeze(-1)
    print(out.shape)

#transpose()每次只能转两个维度
#permute()允许所有维度进行重排
def f03():

    tensor1 = torch.rand(3,4,5)

    out = tensor1.transpose(1,2)

    print(out.shape)

    tensor2 = torch.rand(3,4,5,1,6,9,8,2,7)
    #如果目标是1,8,4,6,7,3,2,9,5
    out =  tensor2.permute(3,6,1,4,8,0,7,5,2)
    print(out.shape)

def f04():
    tensor1 = torch.randn(2,3,4)

    #将张良进行专职使得百年未不连续的张量
    tensor1 = tensor1.transpose(0,1)
    tensor1 = tensor1.contiguous()
    print(tensor1.is_contiguous())
    result = tensor1.view(6,4)
    print(result.shape)

if __name__ == '__main__':
    # f01()
    # f02()
    # f03()
    f04()