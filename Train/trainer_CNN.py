import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from Train.data_gen import Generator
from tqdm import tqdm


def train(net=None, model_name_and_path='./Models_weights/model',device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
    gen = Generator()
    criterion = nn.L1Loss()
    for epoch in range(1,16):  # loop over the dataset multiple times
        print('epoch:',epoch)
        running_loss = 0.0
        i = 0
        # decreasing the learning rate after 5 epochs
        if epoch  == 5:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        elif epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001
        for i in tqdm(range(0,3500)):
            # get the inputs; data is a list of [inputs, labels]
            k=0
            file_no = random.randint(0, 60000)
            for k in range (0,16):
                file_no = file_no + 1
                if k == 0 :
                    inputs, labels = gen.get_data(file_no)
                if k > 0 :      
                    inputs_stack, labels_stack = gen.get_data(file_no)
                    inputs = torch.cat([inputs, inputs_stack])
                    labels = torch.cat([labels, labels_stack])
            
            INN = inputs.to(device)
            OUT = labels.to(device)
            optimizer.zero_grad()
            outputs = net(INN)
            loss = criterion(outputs, OUT)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            LOSS = 0.0
            if i == 3499:    # print every 2000 mini-batches# CHANGE THIS VALUE
                print('[%d, %5d] loss: %.5f' %
                    (epoch , i + 1, running_loss / 3500))
                running_loss = 0.0
                g = 0
                for g in range(0,750):
                            k=0
                            file_no = random.randint(60012, 72730)
                            for k in range (0,16):
                                file_no = file_no + 1
                                if k == 0 :
                                        inputs, labels = gen.get_data(file_no)
                                if k > 0 :      
                                        inputs_stack, labels_stack = gen.get_data(file_no)
                                        inputs = torch.cat([inputs, inputs_stack])
                                        labels = torch.cat([labels, labels_stack])
                            INN = inputs.to(device)
                            OUT = labels.to(device)
            # zero the parameter gradients
                            optimizer.zero_grad()
            # forward + backward + optimize
                            outputs = net(INN)
                            loss = criterion(outputs, OUT)
                            LOSS  += loss.item()
                LOSS = LOSS/750
                print('Avergae_validation_loss was='+str(LOSS) )
                running_loss = 0.0
                if epoch ==1 :
                    best_loss = LOSS
                    torch.save(net, model_name_and_path +str('.pt'))
                if LOSS < best_loss:
                    best_loss = LOSS
                    torch.save(net, model_name_and_path +str('.pt'))
        
                LOSS = 0    
                running_loss = 0.0   

    print('Finished Training')
