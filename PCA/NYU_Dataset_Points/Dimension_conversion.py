from re import X
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk


def higher_space_to_Lower(path_to_higher_dimesnional = './Higher Dimensional 42 Hand Joints/42_Joints_3D_Points.csv'):
    """
    This function uses Principal component analysis to convert the 3D Coordinates for the 42 hand Joints 
    into lower dimensional 30 points only. These 30 lower dimesional points shall be used at the output of the Convolutional Neural Network 
    as the Ground Truth. 
    The main idea is instead of direct regression to 42 hand joints in the 3D space one must target to regression on the lower dimensional embedding
    of the hand joints.
    This function returns : Lower dimensional dataframe
    input : optional path to higher dimensional data 
    """
    higher_space_points = pd.read_csv(path_to_higher_dimesnional)
    # Experiments were conducted to find the optimal number of components
    # This 30 number also is present in literature to b ethe optimal number
    pca = PCA(n_components=30)
    pca.fit(higher_space_points)
    lower_space_points = pca.transform(higher_space_points)
    lower_space_points_dataframe =  pd.DataFrame(lower_space_points,index=  None)
    pk.dump(pca, open("pca.pkl","wb"))
    return lower_space_points_dataframe



def lower_space_to_higher( Y = None, path_to_pca_file = "pca.pkl"):
    """
    This function shall return the inverse tranform from lower dimesnional embedding to higher dimesnion i.e. 
    3D representation of 42 hand joint.
    """   
    pca = pk.load(open(path_to_pca_file,'rb'))
    if Y == None:
        return pca
    else:
        return pca.inverse_transform(Y)

