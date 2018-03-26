import matplotlib
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pylab as plt
import random


matplotlib.use("Qt5Agg")

# 生成数据

# 输入数量
num_inputs = 2
# 样本数量
num_examples = 1000

# 真实的 w
true_w = [2, -3.14]
# 真实的 b
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b

y += .01 * nd.random_normal(shape=y.shape)

# 使用 matplotlib 观察散点图
# plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
# plt.show()

# 数据读取

# 每次返回样本数量
batch_size = 10


def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


# 读取第一个数据块
# for data, label in data_iter():
#     print(data, label)
#     break

# 初始化模型参数
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

# 使用参数求导来更新参数的值,使损失尽量减小 因此需要在创建参数的梯度
for param in params:
    param.attach_grad()


# 定义模型, 线性模型: 将输入和模型的权重(w) 相乘. 再加上偏移量(b)
def net(X):
    return nd.dot(X, w) + b


# 损失函数, 使用平方差衡量预测目标和真实目标之间的差距
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


# 优化. 使用梯度下降求解线性模型, 使用随机梯度下降来求解.
# 将模型参数沿着梯度的反方向走特定的距离, 这个距离称为学习率(learning rate)
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 训练
# 使用 epoch 表示迭代从此时,
#  在一次迭代中,每次随机读取数个数据点.
#  计算梯度并更新模参数

# 真实的模型函数
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


# 绘制损失随着训练次数讲点的折线图, 已经预测值和正式值的散点图
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f1, (fg1, fg2) = plt.subplots(1, 2, )

    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')

    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(), real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')

    fg2.legend()
    plt.show()


epochs = 5
learning_rate = .001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01

# 训练
for e in range(epochs):
    total_loss = 0

    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss / (1 - (1 - smoothing_constant) ** niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (
                e, niter, est_loss, total_loss / num_examples))
            plot(losses, X)
