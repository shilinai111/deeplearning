import torch

torch.manual_seed(14)

tensor1 = torch.randint(0, 5, (2, 3), dtype=torch.float)
print(f'原始张量:\n{tensor1}')
def f01():
    #均值
    print(tensor1.mean)

    #行均值，列均值
    print(tensor1.mean(dim=0))    #列
    print(tensor1.mean(dim=1))    #行

def f02():
    print(tensor1.sum)

    print(tensor1.sum(dim=0))    #列
    print(tensor1.sum(dim=1))    #行

def f03():

    # print(tensor1.max(dim=0))    #列
    # print(tensor1.max(dim=1))    #行

    #分离 极值和索引
    print(tensor1.min(dim=0).values)
    print(tensor1.min(dim=1).indices)
def f04():
    #开根号
    # tensor1[0,0] = -1
    # print(tensor1.sqrt())

    #幂运算
    print(tensor1.pow(2))
    print(tensor1.pow(-2))

    #指数运算
    print(tensor1.exp())

    #对数运算
    print(tensor1.log())   #e为底
    print(tensor1.log2())  #2为底
    print(tensor1.log10())

if __name__ == '__main__':
    # f01()
    # f02()
    # f03()
    f04()