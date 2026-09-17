import torch

def  f01():
    a = torch.tensor(10,)
    b = torch.tensor(20,)

    # 方法1
    print(a+b)
    print(a - b)
    print(a * b)
    print(a / b)

    #方法2
    print(torch.add(a,b))
    print(torch.sub(a,b))
    print(torch.mul(a,b))
    print(torch.div(a,b))

    # 方法3
    print(a.add(b))
    print(a.sub(a))
    print(a.mul(a))
    print(a.div(a))
    # 方法4

# 矩阵点乘
def f02():
    a = torch.randint(0,5,(2,3))
    b = torch.randint(0, 5,(2, 3))

    print("a\n",a)
    print("b\n",b)
    c = torch.mul(a,b)
    print("c\n",c)

def f03():
    a = torch.randint(0,5,(2,3))
    b = torch.randint(0,5,(3,6))

    out = torch.matmul(a,b)

    print('out\n',out,out.shape)


if __name__ == '__main__':
    # f01()
    f02()
    f03()