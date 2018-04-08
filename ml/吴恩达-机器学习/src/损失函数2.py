import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# h(x) = 0.06x + 50
X = []
y = []
# 模拟数据

true_t1 = 0.06
true_t0 = 50
m = 3

for i in range(m):
    X.append(i)
    y.append(true_t1 * i + true_t0)

#y = nd.array(y)
# # 给y加上噪音 噪音服从均值为零 方差为0.1的正太分布
#y += nd.random_normal(shape=y.shape) * 0.01


# 散点图
# plt.scatter(X, y.asnumpy())
# plt.show()


# 损失函数J
def _j(theta0, theta1):
    sum = 0
    for i, x in enumerate(X):
        sum = sum + (theta0 + theta1 * x - y[i]) ** 2
    return sum / (2 * len(X))


# 初始化参数
t0 = 0.0
t1 = 0.0
learning_rate = 0.01


# 梯度下降
def sgd_t0():
    sum = 0
    for i, x in enumerate(X):
        sum = sum + ((t1 * x + t0) - y[i])
    return t0 - learning_rate * (sum / m)


def sgd_t1():
    sum = 0
    for i, x in enumerate(X):
        sum = sum + ((t1 * x + t0) - y[i]) * x
    return t1 - learning_rate * (sum / m)


epochs = 5000

# 训练
for e in range(epochs):
    temp0 = sgd_t0()
    temp1 = sgd_t1()
    t0 = temp0
    t1 = temp1

print('t0:', t0)
print('t1:', t1)
print(_j(t0, t1))
