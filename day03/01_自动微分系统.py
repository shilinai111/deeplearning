import torch

#定义一个标量赞数张量（浮点），作为模型待优化的参数
#参数require_grad = Ture,表示这个参数开启了梯度计算，一位置被计算图追踪
#并指定参数w的初始值为浮点数10
w = torch.tensor(10,requires_grad=True,dtype = torch.float)

#定义一个取消为loss函数
y = 2* w**2

# 中间张量 y，就算有`grad_fn`，**默认不会保存`.
# grad`**；只有叶子节点才存`.grad`。
# 如果要中间变量的梯度，需要 `y.retain_grad()`
#查看梯度函数类型，即曲线函数类型
print(y.grad_fn)

#反向传播计算梯度（y是一个标量可以不用y.sum() 转成标量）
y.backward()

#打印w梯度参数
print(w.grad)