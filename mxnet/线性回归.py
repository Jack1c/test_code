# 线性回归

import mxnet.ndarray as nd
import mxnet.autograd as autograd
import random

# 模拟数据
num_inputs = 2
num_explems = 1000

true_w = [23, -0.9]
true_b = 33

X = nd.random_normal(shape=(num_explems, num_inputs))
y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b

# 定义参数
w = nd.random_normal(shape=(num_inputs,))
b = nd.zeros(shape=(1,))
params = [w, b]

# 给参数开梯度
for param in params:
    param.attach_grad()


# 优化
def SGD(lr):
    for param in params:
        param[:] = param - lr * param.grad


# 训练模型
def net(X):
    return nd.dot(X, w) + b


# 损失函数
def sequare_loss(y, yhate):
    return (y.reshape(yhate.shape) - yhate) ** 2


# 数据读取
batch_size = 10


def data_iter():
    idx = list(range(num_explems))
    random.shuffle(idx)
    for i in range(0, num_explems, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_explems)])
        yield nd.take(X, j), nd.take(y, j)


# 训练
learning_rate = 0.01
epochs = 5
for e in range(epochs):
    for data, label in data_iter():
        total_loss = 0
        with autograd.record():
            loss = sequare_loss(net(data), label)
            loss.backward()
        SGD(learning_rate)
        total_loss += loss
    print('total_loss', nd.sum(total_loss)[0])

print(params)
