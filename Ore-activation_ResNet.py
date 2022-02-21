# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 00:00:56 2021

@author: Mahmoud Abbasi
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision 
from torchvision import transforms

# define pre-activation residual block
class PreActivationBlock(nn.Module):
    expansion = 1
    def __init__(self, in_slices, slices, stride=1):
        super(PreActivationBlock, self).__init__()
        
        self.bn_1 = nn.BatchNorm2d(in_slices)
        self.conv_1 = nn.Conv2d(in_channels=in_slices,
                                out_channels=slices, kernel_size=3,
                                stride = stride, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(slices)
        self.conv_2 = nn.Conv2d(in_channels=slices,
                                out_channels=slices, kernel_size=3,
                                stride = 1, padding=1, bias=False)
        
        # if the input/output dimensions differ use convolution for the shortcut
        if stride != 1 or in_slices != self.expansion * slices:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_slices, out_channels=self.expansion * slices,
                          kernel_size=1, stride=stride, bias=False)
                )
    
    def forward(self, x):
        out = F.relu(self.bn_1(x))
        
        # reuse bn+relu in downsampling layers
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv_1(out)
        
        out = F.relu(self.bn_2(out))
        out = self.conv_2(out)
        
        out += shortcut
        
        return out
        
####### the bottleneck version of the residual block

class PreActivationBottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_slices, slices, stride=1):
        super(PreActivationBottleneckBlock, self).__init__()
        
        self.bn_1 = nn.BatchNorm2d(in_slices)
        self.conv_1 = nn.Conv2d(in_channels=in_slices, out_channels=slices,
                                kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(slices)
        self.conv_2 = nn.Conv2d(in_channels=slices, out_channels=slices,
                                kernel_size=3, stride=stride, padding=1,
                                bias=False)
        self.bn_3 = nn.BatchNorm2d(slices)
        self.conv_3 = nn.Conv2d(in_channels=slices, out_channels=self.expansion*slices,
                                kernel_size=1, bias=False)
        
        # if the input/output dimensions differ use convolution for the shortcut
        if stride != 1 or in_slices != self.expansion*slices:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_slices, out_channels=self.expansion*slices,
                          kernel_size=1, stride=stride, bias=False)
                )
    def forward(self, x):
        out = F.relu(self.bn_1(x))
        
        # reuse bn+relu in downsampling layers
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv_1(out)
        
        out = F.relu(self.bn_2(out))
        out = self.convbn_2(out)
        
        out = F.relu(self.bn_3(out))
        out = self.convbn_3(out)
        
        out +=shortcut
        
        return out
    
    
####### Residual network itself
class PreActivationResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        :param block: type of residual block (regular or bottleneck)
        :param num_blocks: a list with 4 integer values.
        Each value reflects the number of residual blocks in
       the group
       :param num_classes: number of output classes
        """
        super(PreActivationResNet, self).__init__()
        self.in_slices = 64
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_1 = self._make_group(block, 64, num_blocks[0], stride=1)
        self.layer_2 = self._make_group(block, 128, num_blocks[1], stride=2)
        self.layer_3 = self._make_group(block, 256, num_blocks[2], stride=2 )
        self.layer_4 = self._make_group(block, 512, num_blocks[3], stride=2 )
        self.linear = nn.Linear(in_features=512*block.expansion,
                                out_features=num_classes)


    def _make_group(self, block, slices, num_blocks, stride):
        # Create one residual group
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_slices, slices, stride))
            self.in_slices = slices * block.expansion
    
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
        
 
# ResNet configuration with with 34 convolution layers \
#grouped in [3, 4, 6, 3] non-bottleneck residual blocks
def PreActivationResNet34 ():
    return PreActivationResNet(block=PreActivationBlock,num_blocks=[3, 4, 6, 3])
        
        
####################### training the network        

# train data transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))
    ])    
        
# train data loader
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True,
                                         transform=transform_train)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100,
                                           shuffle=True)

# test data transform
        
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))
    ])            
        
   
# test data loader
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True,
                                         transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=100,
                                           shuffle=True)


########## Instantiate the network model and the training parameters

# load pretrained model
model = PreActivationResNet34()

# select our device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # transfer the model to the device
model = model.to(device)

# loss function
loss_function = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters())

# Function for training and testing of the model
def train_model(model, loss_function, optimizer, data_loader):
    model.train()
    current_loss = 0.0
    current_acc = 0
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = inputs.to(labels)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            labels = labels.squeeze(1)
            loss = loss_function(outputs, labels)
            
            #  backward pass
            loss.backward()
            optimizer.step()
            
            # statistics
            current_loss += loss.item() * inputs.size(0)
            current_acc += torch.sum(predictions == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    
    print(f'Train Loss: {total_loss}:.4f; Accuracy: {total_acc}:.4f')
 


def test_model(model, loss_function, data_loader):
    model.eval()
    
    current_loss = 0.0
    current_acc = 0
    
    for i, (inputs, labels) in enumerate(data_loader):
        # send inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward pass
        with torch.set_grad_enabled(False):
            outputs = model (inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
            
        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    
    print(f'Test Loss: {total_loss}:.4f; Accuracy: {total_acc}:.4f')
    
    return total_loss, total_acc
  
   

EPOCHS = 15

# collect accuracy for plotting
test_acc = list()

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1} / {EPOCHS}')
    
    train_model(model, loss_function, optimizer, train_loader)
    _, acc = test_model(model, loss_function, optimizer, test_loader)
    test_acc.append(acc)
    
print(f'test accuracy is {test_acc}')
    
    























     
       
