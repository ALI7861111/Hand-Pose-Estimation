import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# MODEL FROM THE RESEARCH PAPER
class Net_old(nn.Module):

    def __init__(self):
        super(Net_old, self).__init__()
        # kernel
        # (32-5/1)+1 = ouput of each filter
        # output all filters = 28 x92 = 2576 
        self.conv1 = nn.Conv3d(1,96,(5,5,5), padding=0)
        self.pool  = nn.MaxPool3d(2, stride=2)
        # you can use this formula [(W−K+2P)/S]+1.
        self.conv2  =  nn.Conv3d(96,192,(3,3,3), padding=0)
        self.pool1  =  nn.MaxPool3d(2, stride=2)
        self.conv3  =  nn.Conv3d(192,384,(3,3,3), padding=0)
        self.pool2  =  nn.MaxPool3d(2, stride=2)
        self.ln1    =  nn.Linear(in_features=3072, out_features=4096)
        self.ln2    =  nn.Linear(in_features=4096, out_features=1024)
        self.out1    =  nn.Linear(in_features=1024, out_features=30)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = x.view(-1, 3072)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.relu(x)
        x1 = self.out1(x)
        ret = x1.view(-1, 30)
        return ret

    def num_flat_features(self, x):
        x = self.x
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    