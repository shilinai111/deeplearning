from torch import nn
import matplotlib.pyplot as plt


def f01():
    linear = nn.Linear(128, 256)

    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)

    # nn.init.uniform_(linear.weight)
    # nn.init.uniform_(linear.bias)

    # print(linear.weight.data,linear.weight.data,linear.bias.data.shape)
    # print(linear.bias.data,linear.bias.data.shape)

    plt.hist(linear.weight.data.flatten().numpy(), bins=100)
    plt.show()

def f02():
    linear = nn.Linear(128, 256)

    nn.init.constant_(linear.weight, 3)
    nn.init.constant_(linear.bias, 4)

    print(linear.weight.data)
    print(linear.bias.data)


def f03():
    linear = nn.Linear(128, 256)

    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

def f04():
    linear = nn.Linear(128, 256)

    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

def f05():
    linear = nn.Linear(128, 256)

    nn.init.kaiming_normal_(linear.weight)

    # nn.init.kaiming_uniform_(linear.weight)

    print(linear.weight.data)
    print(linear.bias.data)


    plt.hist(linear.weight.data.flatten().numpy(), bins=30)
    plt.show()

def f06():
    linear = nn.Linear(128, 256)

    nn.init.xavier_normal(linear.weight)

    # nn.init.xavier_uniform(linear.weight)

    print(linear.weight.data)
    print(linear.bias.data)


    plt.hist(linear.weight.data.flatten().numpy(), bins=30)
    plt.show()



if __name__ == '__main__':
    # f01()
    # f02()
    # f03()
    # f04()
    # f05()
    f06()