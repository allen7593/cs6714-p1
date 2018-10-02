import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Regression
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())
#
# x, y = Variable(x), Variable(y)
#
#
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
#
# net = Net(1, 10, 1)
# print(net)
#
# plt.ion()
# plt.show()
#
# optimozer = torch.optim.SGD(net.parameters(), lr=0.4)
#
# loss_func = torch.nn.MSELoss()
#
# for t in range(100):
#     prediction = net(x)
#
#     loss = loss_func(prediction, y)
#
#     optimozer.zero_grad()
#     loss.backward()
#     optimozer.step()
#
#     if t % 5 == 0:
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, "Loss=%0.4f" % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()

# classification
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
print(net)

plt.ion()
plt.show()

optimozer = torch.optim.SGD(net.parameters(), lr=0.02)

loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimozer.zero_grad()
    loss.backward()
    optimozer.step()

    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
