# Hand-Pose Estimation Using Depth Images

We aim to modify the work done by Liuhao Ge and produce better optimizations to find the optimal point between accuracy and latency. Our benchmarking have proved that our proposed archhitecture performs better.


*Right side shows the Prediction of Hand-Joints. 
Left-side shows the point cloud representation of deph images*
![ Joint Prediction from TSDF ](Docs\res\Prediction.gif)  

## Depth Image to Volumetric Conversion

![ Joint Prediction from TSDF ](Docs\res\Depth_to_TSDF.PNG) 

### Depth Image
In our project depth images from the NYU dataset were used for conversion into volume. The depth images were collected from Microsoft Kinect V2. The link to the open source dataset is given below:

https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm 

### Point Cloud

The depth image does have 3D information but from one field of view. This depth of a pixel can be interpreted as an extra dimension for the point cloud. This dimension included describes each pixel with discrete point values in R^3space. The figure included demonstrates this depth description with a reasonable degree of accuracy. The color of the visualization is based on the value of the Z-axis (depth) with blue being close to the camera and red being further away.


### Axis Aling Bounding Box (AABB) and Voxel Space for TSDF

AABBs are extensively used as a primitive in 3D graphics rendering. AABBs have only two attributes both being in 3D space. The First defines a minimum bound and the second defines the maximum bound. Only having two attributes for a point cloud bound can make AABBs really computation efficient. Traditionally this primitive is used for object intersection to find if an object is behind another object or overlapping if at all, but this work is not using this primitive in that manner. This geometry is instead used to define an outline for the voxel space for our TSDF. The figure shows the AABB as well as the voxel grid outline. The voxel grid will always be a cube to avoid dynamic voxel values on each frame


### Occupany Grid

Occupancy Grid is a binary grid in 3D space, representing the voxels that have a point in it as ones and the voxels that are empty as zeros. For this work, a volume resolution value of 32 is being used. This implies that each volume has 32x32x32 voxels. This can be higher but it is observed to be highly computationally expensive with no discernable accuracy improvement. The increase in computation cost is due to the fact that the algorithm scales with a big O notation of O(n3) where n being the volume resolution value

### TSDF

A Signed Distance Field (SDF) of a 3D space stores a distance value in each voxel and is found out by calculating the distance from the voxel center and each point in the point cloud and storing the minimum distance value in the voxel. The voxels in front of the boundary of the point have different signs than the voxels that are inside the point cloud. This value increases as the voxels move further away from the point. The value increases indefinitely without bounds if the voxels are too far away. This can be mitigated by applying a truncation value condition on the signed distance function and that’s where the name of the truncated signed distance function comes from. The truncation value determines how big the values have to be, to be assigned as unity


### *How to Create TSDF from Depth Images*
1. Download all the depth images put them into a single folder. The depth images used for this project were taken from NYU dataset. 

``` python
from tools.datamanager import Batch
from tools.datamanager import TSDFVisualization
import cv2
import numpy as np

# The commDepth varaible is for selecting files starting with specific name. It is useful if you have multiple files in the folder.
# In the NYU folder there are also RGB images but Depth images have a specific starting name. 
test = Batch(batchName='test', commDepth='synthdepth_1_')
# test.getDepth('path to folder with depth images')
test.getDepth('nyu\\trainrandom\\synth')
# The function below visualizes the TSDF created. The input argument is the image/TSDF number to be visualized 
TSDFVisualization(test.getAccurateTSDF(33))
# The function below is used to created TSDF in .h5 files. The files shall be saved in the TSDF folder
test.makeAccurateTSDF()
```
The TSDF have already been created 


### *How to Train the model on TSDF*

#### 1. Training the Ge Le (2019) architecture

```python

from Train.model_old_Ge_Le import Net_old
from Train.trainer_CNN import train

model = Net_old()
# Please do not include the .pt extension in the name
train(net=model),  model_name_and_path= 'path/name_of_model' )
```
#### 2. Training our modified architecture.

```python

from Train.model_new_modified import class Net_modified
from Train.trainer_CNN import train

model = Net_modified()
# Please do not include the .pt extension in the name
train(net=model,  model_name_and_path= 'path/name_of_model' )

```


## $Refrences$:

Real-time 3D Hand Pose Estimation
with 3D Convolutional Neural Networks

*Liuhao Ge, Hui Liang, Member, IEEE, Junsong Yuan, Senior Member, IEEE, and Daniel Thalmann*

A. Haque, B. Peng, Z. Luo, A. Alahi, S. Yeung, and F.-F. Li, “Towards
viewpoint invariant 3D human pose estimation,” in Proc. Eur. Conf.
Comput. Vis., 2016, pp. 160–177