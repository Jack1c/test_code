import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# h(x) = 0.06x + 50
X = []
y = []
# 模拟数据

for i in range(3):
    X.append(i)
    y.append(i * 0.06 + 50)

# y = nd.array(y)
# # 给y加上噪音 噪音服从均值为零 方差为0.01的正太分布
# for i, n in enumerate(nd.random_normal(shape=y.shape) * .001):
#     y[i] = y[i] + n

# 散点图
# plt.scatter(X, y.asnumpy())
# plt.show()

# y = y.asnumpy()


#
def _j(theta0, theta1):
    sum = 0
    for i, x in enumerate(X):
        sum = sum + (theta0 + theta1 * x - y[i]) ** 2
    return sum / (2 * len(X))


# j相对于theat0(-0.5~0.5) 和theat1(-1000~2000) 的函数图
fig = plt.figure()
ax = Axes3D(fig)

theat0 = []
theat1 = []
Jt0t1 = []
for i in range(-20, 10):
    if i % 2 == 0:
        for j in range(-20, 10, 1):
            if j  % 2 == 0:
                theat0.append(i)
                theat1.append(j)
                Jt0t1.append(_j(i, j))

theat0, theat1 = np.meshgrid(theat0, theat1)

Jt0t1 = np.array(Jt0t1)

print(theat0.shape)
print(theat1.shape)
print(Jt0t1.shape)
ax.plot_surface(theat0, theat1, Jt0t1, rstride=1, cstride=1, cmap='rainbow')


