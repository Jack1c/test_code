import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
import random

# 创建数据集
num_inputs = 2
num_examples = 1000

true_w = [2, -3.14]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b

y += 0.01 * nd.random_normal(shape=y.shape)

batch_size = 10


# 数据读取
def data_iter():
    idx = list(range(num_examples))
    # 将索引序列打乱
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


# for data, label in data_iter():
#     print(data, label)
#     break

# 初始化参数
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros([1, ])
params = [w, b]

# 创建参数的梯度
for param in params:
    param.attach_grad()


# 定义模型
def net(X):
    return nd.dot(X, w) + b


# 损失函数
def sequare_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


# 优化
def SDG(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 训练
epochs = 5
learning_rate = 0.01
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            loss = sequare_loss(net(data), label)
        loss.backward()
        SDG(params, learning_rate)
        total_loss += loss
    print(nd.sum(total_loss))

print(w, b)
