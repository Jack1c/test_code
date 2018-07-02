import sys
sys.path.append("..")
import gluonbook as gb
from mxnet import autograd,gluon, init, nd
from mxnet.gluon import loss as gloss, nn

batch_size = 1024
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Flatten())
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

learin_rate=0.01

num_epochs = 200
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             learin_rate, trainer)
