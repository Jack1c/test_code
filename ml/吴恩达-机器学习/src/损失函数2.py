import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# h(x) = 0.06x + 50
X = []
y = []
# 模拟数据

for i in range(30):
    X.append(i)
    y.append(i * 0.06 + 50)

y = nd.array(y)
# # 给y加上噪音 噪音服从均值为零 方差为0.1的正太分布
y += nd.random_normal(shape=y.shape) * 0.1

# 散点图
# plt.scatter(X, y.asnumpy())
# plt.show()


# 损失函数J
def _j(theta0, theta1):
    sum = 0
    for i, x in enumerate(X):
        sum = sum + (theta0 + theta1 * x - y[i]) ** 2
    return sum / (2 * len(X))

# 梯度下降



