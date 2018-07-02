import numpy as np

# 模拟数据
n = 5
true_w = np.array([4.2, 2, -3.4])
X = np.random.randint(1, 10, len(true_w) * n).reshape((n, len(true_w)))
X.transpose()[0] = 1

y = np.empty(shape=(n))


def net(X, w):
    return np.dot(X, w)


for i, _x in enumerate(X):
    y[i] = net(_x, true_w)

# 数据读取


# 初始化参数
epochs = 50000
learning_reate = 0.01
w = np.zeros(len(true_w))


# 梯度下降
def sdg(j):
    sum = 0
    for i, x in enumerate(X):
        sum = sum + (net(x, w) - y[i]) * x[j]
    return sum / n


# 训练
for i in range(epochs):
    temp = np.zeros(len(true_w))
    for j, t in enumerate(w):
        temp[j] = t - learning_reate * sdg(j)
    w = temp
print(true_w)
print(w)
