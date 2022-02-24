from torch import nn


class Classify(nn.Module):
    def __init__(self,inputdimension,outdim):
        super(Classify,self).__init__()
        self.in_lier = nn.Linear(inputdimension,10)
        self.hidden0 = nn.Linear(10,20)
        self.hidden1 = nn.Linear(20,30)
        self.output  = nn.Linear(30,outdim)
        self.sigmoid = nn.Sigmoid()
