from torchvision import datasets, transforms
import torch
from torch import _test_serialization_subcmul, nn
import torch.nn.functional as F
from model import Classify
from torch import optim
import matplotlib.pyplot as plt
from helper_functions import acc


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

epochs=50
losses = []
test_losses = []

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

    print('Train Loss:  ', running_loss/len(trainloader))
    losses.append(running_loss/len(trainloader))
    clf.eval()
    with torch.no_grad():
        test_running_loss = 0
        for images_test, labels_test in iter(testloader):

            images_test = images_test.view(images_test.shape[0], -1)
            test_pred = clf.forward(images_test)
            test_loss = criterion(test_pred, labels_test)
            test_running_loss += test_loss.item()

        print('Test Loss:   ', test_running_loss/len(testloader))
        test_losses.append(test_running_loss/len(testloader))
        if e % 5 == 0:
            print('Accuracy', acc(test_pred,labels_test))
    clf.train()
plt.plot(losses, label='trainlosses')
plt.plot(test_losses, label='testlosses')   
plt.legend()
plt.show()
