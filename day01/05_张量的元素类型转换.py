import torch
def f01():
    out = torch.rand(3,5)
    print(out,out.dtype)

    out2 = out.type(torch.float64)

    print(out2,out2.dtype)

def f02():
    out = torch.rand(3,5)
    print(out.half().dtype)
    print(out.float().dtype)
    print(out.double().dtype)
    print(out.short().dtype)
    print(out.int().dtype)
    print(out.long().dtype)

if __name__ == '__main__':
    f01()
    f02()