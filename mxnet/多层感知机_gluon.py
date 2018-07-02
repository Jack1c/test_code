import sys

sys.path.append("..")
import gluonbook as gb
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(nn.Dense(128, activation='tanh'))
net.add(nn.Dense(10))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_szie = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_szie)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {
    'learning_rate': 0.1
})
epochs = 50
gb.train_cpu(net,train_iter,test_iter,loss,epochs,batch_szie,trainer=trainer)


