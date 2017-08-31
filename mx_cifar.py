from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

from tqdm import tqdm
import load_cifar

batch_size = 256

class Cifar10Dataset(gluon.data.dataset.Dataset):
    """Base class for MNIST, cifar10, etc."""
    def __init__(self, dataset):
        self._data, self._label = dataset

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)

trainset, testset = load_cifar.load()

train_data = gluon.data.DataLoader(
    Cifar10Dataset(trainset),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    Cifar10Dataset(testset),
    batch_size=batch_size, shuffle=False, last_batch='discard')



vgg_net = gluon.nn.Sequential()
with vgg_net.name_scope():
    vgg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    for x in vgg:
        if x == 'M':
            vgg_net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        else:
            vgg_net.add(gluon.nn.Conv2D(channels=x, kernel_size=3, padding=1))
            vgg_net.add(gluon.nn.BatchNorm(momentum=0.1))
            vgg_net.add(gluon.nn.LeakyReLU(0))
#            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                       nn.BatchNorm2d(x),
#                       nn.ReLU(inplace=True)]
#    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    vgg_net.add(gluon.nn.AvgPool2D(pool_size=1, strides=1))
    vgg_net.add(gluon.nn.Flatten())
#    vgg_net.add(gluon.nn.Dense(4096))
    vgg_net.add(gluon.nn.Dense(10))
ctx = mx.gpu()
vgg_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print(vgg_net)

initial_lr = 0.01
trainer = gluon.Trainer(vgg_net.collect_params(), 'sgd',
                {'learning_rate': initial_lr, 'momentum':0.9, 'wd':5e-4})
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    total_loss=0
    for d, l in tqdm(data_iterator):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        loss = softmax_cross_entropy(output, label)
        total_loss += nd.mean(loss).asscalar()
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return total_loss/len(data_iterator), acc.get()[1]
#evaluate_accuracy(test_data,vgg_net)
###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################

import time

initial_lr = 0.01
trainer = gluon.Trainer(vgg_net.collect_params(), 'sgd', {'learning_rate': initial_lr})
best_acc=0
start_epoch=0
timer = 0
for epoch in range(start_epoch,10):  # loop over the dataset multiple times
    start=time.time()
    train_loss = 0.0
    for i, (d, l) in enumerate(tqdm(train_data)):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = vgg_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        train_loss += nd.mean(loss).asscalar()

    test_loss, test_accuracy = evaluate_accuracy(test_data, vgg_net)
    train_loss = train_loss / len(train_data)
    batch_time = (time.time()-start)/len(train_data)
    timer += time.time()-start
    print('Epoch %3d: time: %5.2fs, loss: %5.4f, test loss: %5.4f, accuracy : %5.2f %%, %.3fs/epoch, %.3fms/batch' %
          (epoch, timer, train_loss, test_loss, test_accuracy*100, time.time()-start, batch_time))
