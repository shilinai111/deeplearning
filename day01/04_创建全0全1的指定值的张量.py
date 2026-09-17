import torch

# 创建全0张量
def f01():
    out = torch.zeros(5,5)
    print(out,out.shape, out.dtype)

    #根据指定的张量，依照其尺寸，创建全新全0 张量
    tem = torch.randn(100,200)
    out2 = torch.zeros_like(tem)
    print(out2, out2.shape, out2.dtype)

    # 根据指定的张量，依照其尺寸，创建全新全1 张量
    tem = torch.randn(100, 200)
    out3 = torch.ones_like(tem)
    print(out3, out3.shape, out3.dtype)

#torch.fill 基于现有的张量，对其原地修改，将每个元素变为指定值
#torch.full 重新创建张量。让其值变为指定值
def f03():
    out = torch.full((5,5),99)
    print(out,out.shape, out.dtype)

    tem = torch.randn(100, 200)
    out2 = torch.full_like(tem,-99)
    print(out2, out2.shape, out2.dtype)

if __name__ == '__main__':
    # f01()
    f03()