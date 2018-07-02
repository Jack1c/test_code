import mxnet.autograd as autograd
import mxnet.ndarray as nd
import random

# 生成数据
num_input = 2
num_examples = 1000

true_w = [1.9, -3.4]
true_b = 5.6

X = nd.random_normal(shape=(num_examples, num_input))
y = X[:, 0] * true_w[0] + X[:, 1] * true_w[1] + true_b

# 数据读取

# 每次读取数据条数
batch_size = 10


def data_iter():
    # id创建序列
    idx = list(range(num_examples))
    # 将序列打乱
    random.shuffle(idx)
    # 读取数据
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


# 初始化参数
w = nd.random_normal(shape=(2,))
# 要给参数b开梯度所以b也要放到ndarray中
b = nd.zeros(shape=(1,))

params = [w, b]
# 创建参数的梯度
for param in params:
    param.attach_grad()


# 定义模型
def net(X):
    print(X)
    return nd.dot(X, w) + b


# 损失函数
def sequar_loss(yhate, y):
    return (yhate - y) ** 2


# 优化
def SDG(params, learning_rate):
    for param in params:
        param[:] = param - learning_rate * param.grad



# 训练
epochs = 5
learning_reta = 0.01
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with autograd.record():
            loss = sequar_loss(net(data), label)
            loss.backward()
        SDG(params, learning_reta)
        total_loss += loss
    print(nd.sum(total_loss)[0])

print(true_w, true_b)
print(w, b)
