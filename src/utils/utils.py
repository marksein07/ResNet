import torch
from torch import nn
from torch import optim
import torch.utils.data as Data

import torchvision
from torchvision import transforms

import numpy as np

from tqdm import tqdm

import matplotlib

import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def channel_normalization( inputs, stddev = 1e-2, channel_dim = [2,3] ) :
    return ( inputs - inputs.mean(dim=channel_dim, keepdim=True) ) \
            / inputs.std(dim=channel_dim, keepdim=True) * stddev
    
def epoch_train( model, optimizer, criterion, dataloader, device, transformation ) :
    running_loss = 0.0
    accuracy_list = []
    #semantic_loss_list = []
    model.train()
    for i, data in tqdm(enumerate(dataloader, 1)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if transformation :
            inputs = transformation(inputs)
        if device :
            inputs = inputs.to(device)
            labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        ys = outputs.argmax(dim = 1).detach()
        result = torch.eq(ys,labels)
        accuracy_list.append(result.type(torch.FloatTensor).cpu().mean().item())
        Loss = criterion(outputs, labels)
        Loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += Loss.detach().cpu().item()
    return running_loss / i, np.mean(accuracy_list)
        
def test( model, criterion, dataloader, device, transformation ) :
    with torch.no_grad():
        running_loss = 0.0
        accuracy_list = []
        model.eval()
        for j, data in tqdm( enumerate(dataloader, 1) ):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if transformation :
                inputs = transformation(inputs)
            if device :
                inputs = inputs.to(device)
                labels = labels.to(device)
            # zero the parameter gradients

            # forward
            outputs = model(inputs)
            ys = outputs.argmax(dim = 1).detach()
            result = torch.eq(ys, labels)
            accuracy_list.append(result.type(torch.FloatTensor).cpu().mean().item())
            Loss = criterion(outputs, labels).detach().cpu().item()

            running_loss += Loss
    return running_loss / j, np.mean(accuracy_list)

def train(model, 
          optimizer, 
          criterion, 
          trainloader, 
          testloader, 
          device, 
          normalization, 
          writer,
          opt,
          LR,
          epoch = 0, 
          max_epoch = 300, ) :
    epoch_duration = 1
    training_error_rate_list = []
    testing_error_rate_list = []
    training_loss_list = []
    testing_loss_list = []
    for epoch in range(epoch, max_epoch):  # loop over the dataset multiple times

        if epoch > 1 and (epoch % 150 == 0 or epoch % 225 == 0) :
            LR/=10;
            optimizer = opt(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
            print("Optimizer updated.")
        training_loss, training_acc = epoch_train(model, optimizer, criterion, trainloader, device, normalization)

        print('[%d, ] Training loss: %.4f    Training Accuracy : %.4f' % 
              (epoch + 1, 
               training_loss, 
               training_acc, ) ) 
        testing_loss, testing_acc = test(model, criterion, testloader, device, normalization)
        print('[%d, ]  Testing loss: %.4f    Testing  Accuracy : %.4f\n' % 
              ( epoch + 1, 
               testing_loss, 
               testing_acc, ) )
        writer.add_scalars('cifar10/Accuracy', 
                           {'Training' : training_acc,
                            'Testing'  : testing_acc, 
                           }, epoch + 1)
        writer.add_scalars('cifar10/loss', 
                           {'Training' : training_loss, 
                            'Testing'  : testing_loss,  
                           }, epoch + 1)
        training_error_rate_list.append(1-training_acc)
        testing_error_rate_list.append(1-testing_acc)
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)

    print('Finished Training')
    return training_error_rate_list, testing_error_rate_list, training_loss_list, testing_loss_list

def dataloader(BATCH_SIZE, download=True, shuffle=True, augmentation=False):
    normal_transformations = transforms.Compose( [transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=True, download=download, transform=normal_transformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=False, download=download, transform=normal_transformations)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
    if augmentation :
        transformation = [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(p=1.0),
        ]

        augmentation_transformation = transforms.Compose( [
            transforms.RandomChoice(transformation),
            transforms.ToTensor(),
        ] )

        augmentation_trainset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=True, download=download, transform=augmentation_transformation)
        augmentation_trainloader = torch.utils.data.DataLoader(augmentation_trainset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
        return trainloader, testloader, augmentation_trainloader


    return trainloader, testloader
def plot(data_points_dict, filename="plot", log_scale=True, xlabel='epoch', ylabel='error', title='title') :
    if log_scale :
        plt.yscale('log')
    for label, data_points in data_points_dict.items() :
        plt.plot(data_points, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()