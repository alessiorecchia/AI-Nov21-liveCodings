from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from model import Classify
from torch import optim


clf=Classify(28*28,10)



# Define a transform to normalize the data (Preprocessing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

# Download and load the training data
trainset    = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(clf.parameters(),lr=0.003)

epochs=10

for e in range(epochs):
    running_loss=0

    for images, labels in iter(trainloader):
        
        #images.resize(images.size()[0],784)
        images=images.view(images.shape[0],784)
        
        optimizer.zero_grad()
        pred=clf.forward(images)
        loss=criterion(pred,labels)
        loss.backward()
        optimizer.step()


        running_loss+=loss.item()

    print('loss', running_loss/len(trainloader))
