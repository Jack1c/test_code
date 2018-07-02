import sys

sys.path.append("..")
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss

# 获取数据
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

# 定义模型参数
num_inputs = 784
num_outputs = 10
num_hiddens1 = 1024
num_hiddens2 = 1024

W1 = nd.random_normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)

W3 = nd.random_normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b3 = nd.zeros(num_hiddens2)

W2 = nd.random_normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b2 = nd.zeros(num_outputs)

params = [W1, b1, W3, b3, W2, b2]

for param in params:
    param.attach_grad()


# 定义激活函数
def relu(X):
    return nd.maximum(X, 0)


# 定义模型 使用 reshape 将原始图片改为长度为num_inputs的向量
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(nd.dot(X, W1) + b1)
    H2 = relu(nd.dot(H1, W3) + b3)
    return nd.dot(H2, W2) + b2


# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 训练模型
num_epochs = 100
lr = 0.1
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
