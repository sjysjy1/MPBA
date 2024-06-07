from LeNet5 import Model_LeNet5,LeNet5_advertorch
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import torchvision
import torchvision.datasets as datasets
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                #transforms.Normalize((0.1307,), (0.3081,)) #necessary???
                                ])
train_dataset = datasets.MNIST('..//data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=False, transform=transform)
#test_dataset=datasets.MNIST(root='../data',train=False,download=False,transform=transforms.ToTensor())
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=256,shuffle=False)


def training(Network,epoch_num):
    Network.train()

    for epoch in range(epoch_num):
        print('epoch:',epoch)
        for batch_idx,(data,label) in enumerate(train_loader):
            data=data.to(device)
            label=label.to(device)
            output=Network(data)
            loss_eva=loss(output,label) #how does this function know loss????
            #loss_eva = F.nll_loss(output, label)
            loss_eva.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx%10==0:
               print('loss_eva is:',loss_eva.item())
    loss_list.append()
    accuracy_list.append(test(Network))

def test(Network):
    Network.eval()  #how does this function know Network????
    correct_num=0
    with torch.no_grad():
        for data,label in test_loader:# how does this function know the test_loader???
            data=data.to(device)
            label=label.to(device)
            output=Network(data)
            label_predict=torch.argmax(output,1)
            correct_num +=(label==label_predict).sum()
            #correct_num+=torch.equal(label,label_predict).sum()
    print('test accuracy is:',correct_num/10000.0)
    return correct_num/10000.0

seed_torch()
MNIST_model=Model_LeNet5()
MNIST_model.to(device)
learning_rate=0.01
momentum=0.9
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=MNIST_model.parameters(),lr=learning_rate)
training(MNIST_model,epoch_num=10)
test(MNIST_model)
torch.save(MNIST_model, './LeNet5_advertorch_whole.pth')
torch.save(MNIST_model.state_dict(), './LeNet5_advertorch_onlyweight.pth')
