import sys

sys.path.append('..')
import matplotlib.pyplot as plt
from mxnet import autograd, nd
from mxnet.gluon import data as gdata


# 获取数据集
def transform(feature, label):
    return feature.astype('float32') / 255, label.astype('float32')


mnist_train = gdata.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gdata.vision.FashionMNIST(train=False, transform=transform)


# 将数字标签转成相应的文本标签
def get_text_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


# 显示图片
def show_fashion_imgs(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


# X, y = mnist_train[0:9]
# show_fashion_imgs(X)
# print(get_text_labels(y))

# 数据读取 使用gluon.data
batch_size = 256
train_data = gdata.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gdata.DataLoader(mnist_train, batch_size, shuffle=False)

# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)

params = [W, b]

# 给参数附上梯度
for param in params:
    param.attach_grad()


# 定义模型 使用softmax函数将输入值归一化为合法概率
def softmax(X):
    exp = nd.exp(X)
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# 交叉熵的损失函数
def cross_entropy(yhate, y):
    return -nd.pick(nd.log(yhate), y)


# 计算精度
def accuracy(output, label):
    return (output.argmax(axis=1) == label).mean().asscalar()


# 评估精度
def evaluate_accuracy_cpu(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


#
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# print(evaluate_accuracy_cpu(mnist_test, net))

# 训练
num_epochs = 100
lr = 0.1
loss = cross_entropy


def train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
              lr=None, trainer=None):
    for epoch in range(num_epochs):
        trainer_loss_sum = 0
        trainer_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            trainer_loss_sum += l.mean().asscalar()
            trainer_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy_cpu(mnist_test, net)
        print("epoch %d, loss %f, train acc %f, test acc %f"
              % (epoch, trainer_loss_sum / len(train_iter),
                 trainer_acc_sum / len(train_iter), test_acc))


train_cpu(net, train_data, test_data, loss, num_epochs, batch_size, lr)

# 预测

data, label = mnist_test[0:9]
print('true labels')
print(get_text_labels(label))
predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
