import torch

'''
需求:
    #求y=w**2+20 的极小值点并打印y是最小值时w的值(梯度)
    #1 定义张量w=10.,以及学习率
    利用梯度下降法循环迭代1000 求最优解
    #2.1正向计算(前向传播)
    #2.2梯度累加与清零
    #2.3 反向传播
            验证梯度累加 data= w.data- lr *w.grad
    #2.4进一步梯度更新w.
'''

#定义模型参数 张量w=10 ，以及学习率

w = torch.tensor(10,requires_grad=True,dtype=torch.float)
lr = 0.01

#利用梯度下降法，迭代1000次求最优解

for i in range(1000):
    y = w**2 + 20

    if w.grad is not None:
        # 梯度累加清零
        w.grad.zero_()

    #单向传播
    y.backward()

    print(i,'  ',w.grad)

    w.data = w.data - lr * w.grad
    print(w.data, y, w.grad)

'''
不能写为 w = w - lr*w.grad 报错

因为等号右侧计算时，会在内存中新建一个地址来存放w-lr*w.grad 的值
然后赋值操作 将w变量指向这个新内存地址。
而原来的内存地址被系统回收了。
但pytorch 计算图计算梯度，还是追踪原来内存地址中的变量
因此会切断计算图，丢失梯度信息。

w = torch.tensor([1.0])  #W指向对象A
W=W-0.5   #右边的w读取对象A，w-0.5创建对象B，左边的w指向B

打印两个w的idid(w)  2540985325424   2540987189360

'''

