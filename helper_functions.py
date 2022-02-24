import torch 
import torch.nn as nn
import torch.nn.functional as F



def acc(logits,y_target):
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    
    accuracy = sum(y_target == pred)/pred.shape[0]
    return accuracy.item()

# t = torch.rand((10,5))
# p = torch.argmax(t,dim=1)
# print(p)
