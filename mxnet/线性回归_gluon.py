# coding: utf-8

# # 线性回归 - 使 用Gluon

# ## 创建数据集

# In[ ]:


from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.14]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)

# 数据读取

# 使用data模块读取数据
batch_size = 10
data_set = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(data_set, batch_size, shuffle=True)

for data, label in data_iter:
    print(data, label)
    break

# 定义模型

net = gluon.nn.Sequential()
# 加入一个Dense层,它唯一必须要定义的参数是输出的节点个数
net.add(gluon.nn.Dense(1))

# 初始化模型参数
net.initialize()

# 使用gluon提供的平方差函数作为损失函数
square_loss = gluon.loss.L2Loss()

# 优化. 通过创建Trainer的实例,并将模型参数传递给他
trainer = gluon.Trainer(net.collect_params(), 'bfgs', {'learning_rate': .1})

# 训练

epochs = 50
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss / num_examples))

# 先从net中拿到到需要的层, 再获取参数
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())
