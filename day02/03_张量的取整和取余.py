import torch

def f01():
    result1 = torch.tensor([-1.2,0.499,0.5001,1.5,2.5,3.5,4.5,5.5,6.5])

    result2 = result1.ceil()
    print(result2)

    result3 = result1.floor()
    print(result3)

    result4 = result1.round()
    print(result4)

def f02():
    tensor1 = torch.tensor([11,12,13])
    tensor2 = torch.tensor([2,3,4])
    print(tensor1%tensor2)

if __name__ == '__main__':
    f01()
    f02()