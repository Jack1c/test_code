
import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt

# f(x) = 2x + 1
X = []
y = []
# 模拟数据

for i in range(10):
    X.append(i)
    y.append(i * 2 + 1)
y = nd.array(y)
y += nd.random_normal(shape=y.shape) * 2

# 在坐标中显示
plt.scatter(X, y.asnumpy())
plt.show()










