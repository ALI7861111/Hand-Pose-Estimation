import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# MODIFIED MODEL FOR OUR APPROACH
class Net_modified(nn.Module):

    def __init__(self):
        super(Net_modified, self).__init__()
        self.conv1    = nn.Conv3d(1,48,(5,5,5), padding=1)
        self.conv1_1  = nn.Conv3d(1,48,(3,3,3), padding=0)        
        self.pool     =  nn.MaxPool3d(2, stride=2)
        self.conv2    =  nn.Conv3d(96,96,(5,5,5), padding=1)
        self.conv2_2  =  nn.Conv3d(96,96,(3,3,3), padding=0)        
        
        self.pool1    =  nn.MaxPool3d(2, stride=2)
        
        self.conv3    =  nn.Conv3d(192,192,(5,5,5), padding=1)
        self.conv3_1  =  nn.Conv3d(192,192,(3,3,3), padding=0)
        
        self.pool2    =  nn.MaxPool3d(2, stride=2)        
        
        
        self.ln1      =  nn.Linear(in_features=3072, out_features=4096)
        self.ln2      =  nn.Linear(in_features=4096, out_features=1024)
        self.out1     = nn.Linear(in_features=1024, out_features=30)
        
    def forward(self, x):
        #branch one
        x_1 = self.conv1(x)
        x_2 = self.conv1_1(x)
        x_3 = torch.cat([x_2, x_1], dim=1)
        x_4_4 = self.pool(x_3)
        x_4_5 = self.conv2(x_4_4)
        x_4_1 = self.conv2_2(x_4_4)
        x_4 = torch.cat([x_4_1, x_4_5], dim=1)
        x_4 = F.relu(x_4)
        x_4 = self.pool1(x_4)
        x_4_6 = self.conv3(x_4)
        x_4_7 = self.conv3_1(x_4)
        x_4 = torch.cat([x_4_6, x_4_7], dim=1)
        x_4 = F.relu(x_4)
        x_4 = self.pool2(x_4)
        x_4 = x_4.view(-1, 3072)
        x_4 = self.ln1(x_4)
        x_4 = F.relu(x_4)      
        x_4 = self.ln2(x_4)
        x_4 = F.relu(x_4)
        x_4 = self.out1(x_4)
        ret = x_4.view(-1, 30)
         
        return ret

def num_flat_features(x):
    x = x
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features    
    