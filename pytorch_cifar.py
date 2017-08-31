#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 17:37:58 2017

@author: user
"""

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import load_cifar
import torchvision

batch_size = 256
class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data, self.label = dataset

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

trainset, testset = load_cifar.load()

trainloader = torch.utils.data.DataLoader(Cifar10Dataset(trainset), 
                        batch_size=batch_size, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(Cifar10Dataset(testset), 
                        batch_size=batch_size, shuffle=False, num_workers=4)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        vgg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
               512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


net=VGG()
net.cuda()
criterion = nn.CrossEntropyLoss()

print(net)

def test_accuracy():
    correct_n = 0
    total_loss = 0
    for data in tqdm(testloader):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.long().cuda())
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.data[0] * len(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct_n += (predicted == labels.data).sum()
    return total_loss/len(testloader.dataset), 100 * correct_n / len(testloader.dataset)

import time

initial_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
best_acc=0
start_epoch=0
timer = 0
for epoch in range(start_epoch,10):  # loop over the dataset multiple times
    start=time.time()
    train_loss = 0.0
    for data in tqdm(trainloader):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.long().cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.data[0]
    train_loss = train_loss / len(trainloader)
    batch_time = (time.time()-start)/len(trainloader)
    test_loss,accuracy = test_accuracy()
    timer += time.time()-start
    print('Epoch %3d: time: %5.2fs, loss: %5.4f, test loss: %5.4f, accuracy : %5.2f %%, %.3fs/epoch, %.3fms/batch' %
          (epoch, timer, train_loss, test_loss, accuracy, time.time()-start, batch_time))
    

  
print('Finished Training')