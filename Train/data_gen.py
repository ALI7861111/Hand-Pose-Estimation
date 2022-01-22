import numpy as np
import pandas as pd
import h5py
import torch


# This is an comment
class Generator():
    def __init__(self):
      #  pass path to 30 point ground truth
        self.NYU_X = pd.read_csv('./PCA/NYU_Dataset_Points/Lower Dimensional 42 Hand Joints/Lower_Dimension_3D_Points.csv')
    def get_data(self,file_no):
        filename = './TSDF/'+str(file_no)+'.h5'
        h5 = h5py.File(filename,'r')
        input = np.array(h5['TSDF'])
        input = np.reshape(input,(1,1,32,32,32))
        h5.close()           
        inputs = np.array(input).tolist()
        inputs = torch.FloatTensor(inputs)
        output1 = self.NYU_X.iloc[file_no].values
        output1 = output1[0:30]
        output1 = np.asarray(output1)
        output  = torch.from_numpy(output1).float()
        output  = torch.reshape(output, (1, 30))
        return inputs,output


