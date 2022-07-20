# Part Segmentation of Pole-like Objects

This repository contains the implementation of **Kernel Point Convolution** (KPConv) to decompose the pole-like objects to their constituent part (the carrier and different attachments) or as we call it, the part segmentation of the pole-like objects. Based on our custom labeled data, 
The repository is created by Sara Yousefimashhoor as the extension of Hugues THOMAS Deep learning Algorithm (KPConv). <br />
The main changes to the original code are as follows:<br />
•	Debugging the KPConv codes for the part-segmentation task.<br />
•	Adding codes to prepare the custom dataset of the Gemeente Amsterdam for the part segmentation task.<br />
•	Modifying the code to include addtitional features of intensity and color values.<br />



![Intro figure](https://github.com/Amsterdam-Internships/Pole-Part-Segmentation/blob/master/Picture1.png)


## Project Folder Structure
The repository contains the code, dataset, trained model, and some useful resources and examples of the pipeline, which are structured in the following format:

1) [`resources`](./resources): Random nice resources, e.g. [`useful links`](./resources/README.md)
1) [`data`](./data): Folder containing sample data for public use
1) [`KPConv`](./KPConv): Folder containing the code and scripts to run the pipeline
1) [`model`](./model): Folder containing our best trained model for pole part-segmentation task (including coordinate, intensity and color values)
3) [`media`](./media): Folder containing media files (icons, video)
4) ...


## Installation and deployment 

A step-by-step guide to setup the environment for Ubuntu 20.04 is provided in [INSTALL.md](./INSTALL.md). <br />
After setting up the envionment, follow the steps below: <br />

### Training

#### If you are running the pipeline for a custom dataset:  <br />

* Before running the training algorithm, split your data using the stratified split code `0 - stratified_split_for_laz.py` in KPConv folder. This code is reading laz format. Note that for handling `.ply` files requires some modifications.
* Prepare the data using `Prepare_data_xyzrgbi.py` . This code prepares .laz files, saves them as `.ply` with normalized coordinates, color and intensity values and gives them proper names. 
* Create a `Data` folder inside the `KPConv` folder and add the three folders of `train_ply`, `val_ply`, and `test_ply` containing the prepared point clouds to it.
* Modify the `self.path` variable in the `./KPConv/datasets/ShapeNetPart.py` to read the point clouds from the `Data` folder.
* Determine whether the intensity and color values are going to be fed into the model in the `./KPConv/training_ShapeNetPart.py` and `./KPConv/utils/config.py` by modifying the variables `intensity_info` and `color_info`.
* Run the `./KPConv/training_ShapeNetPart.py`

#### If you are runnning on our provided dataset: <br />

* Copy the folders inside the `data` folder to `./KPConv/Data`
* Modify the `self.path` variable in the `./KPConv/datasets/ShapeNetPart.py` folder to read the point clouds from the `Data` folder.
* Determine whether the intensity and color values are going to be fed into the model in the `./KPConv/training_ShapeNetPart.py` and `./KPConv/utils/config.py` by modifying the variables `intensity_info` and `color_info`.
* Run the `./KPConv/training_ShapeNetPart.py`

### Testing
* Find the trained model (log folder_) in `./KPConv/results` that you want to test and copy the path to it.
* Assign the file path to the `chosen_log` variable in `./KPConv/test_any_model.py`
* Make sure that the conda environment `(tf-gpu)` is activated.
* Run `test_any_model.py`

### Inference

## How it works
Kernel-point convolution, or KPConv(Thomas et al., 2019), is a deep learning algorithm operating on 3D point clouds. The algorithm's design is motivated by the concept of having a regular grid for convolution in 2D image processing.  The kernel in this method is a spherical neighborhood with a predefined radius containing a specified number of points on its surface. The arrangement of points on the sphere surface is regular and predefined in the rigid form of KPConv, while in the deformable version of KPConv, the arrangement is learned to fit the local geometry of scene objects. Each kernel point has a learned weight and an area of influence based on Euclidean space and expressed by a correlation function (Thomas et al., 2019). The main assumption is the point cloud's spatial localization property, which allows a spatial kernel to influence a local neighborhood of each point and abstract the neighborhood features. 
To create input for the next layer of the network, The kernel (sphere) center is located on each point in the point cloud. The surrounding points that fell into this neighborhood are considered neighboring points. For each of the neighboring points, a kernel function would be calculated considering its distance from all the kernel points and the learned weights of the kernel points. Each neighboring point influences more from the closest kernel points compared to the farther ones. The calculated kernel function for each neighbor will multiply by its feature matrix and generate a new feature with a lower dimension to describe the center point and its neighborhood as the input to the next layer in the network. <br />

As it can be seen, a flat input containing the points and their corresponding feature(s) is fed to the algorithm. The grid subsampling strategy is used to create a harmonious point density in the point cloud, which makes the network flexible to deal with various point densities. An important hyperparameter is the 'first subsamping distance', which is the first size used for grid subsampling and it is set based on the point spacing in the point cloud.  In each decoding step the number of points is reduced while the subsampling distance and consequently the receptive field area are increased (red balls). In other words, the kernel radius is also updating and enlarging at the same time proportionate to the subsampling distance(Thomas et al., 2019). In the encoding phase, the points are upsampled using the nearest upsampling layer, skip links, and MLP. The ResNet architecture is used to create skip links between encoding layers and the output of decoding layers. The output from the intermediate decoder layer is concatenated with the output of upsampling layers and processed with a shared MLP (unary or 1conv layer) to get the pointwise features(Thomas, 2020; Thomas et al., 2019).In the part segmentation model a single network is trained with multiple heads to distinguish parts of the objects. The point cloud is smaller and multiple instances can be processed simultaneously (Thomas et al., 2019). 





## Parameter Setting 
|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--batch_size`| int| `Batch size.`|  32|
|`--device`| str| `Training device, cpu or cuda:0.`| `cpu`|
|`--early-stopping`|  `store_true`| `Early stopping for training of sparse transformer.`| True|
|`--epochs`| int| `Number of epochs.`| 21|
|`--input_size`|  int| `Input size for model, i.e. the concatenation length of te, se and target.`| 99|
|`--loss`|  str|  `Type of loss to be used during training. Options: RMSE, MAE.`|`RMSE`|
|`--lr`|  float| `Learning rate.`| 1e-3|
|`--train_ratio`|  float| `Percentage of the training set.`| 0.7|
|...|...|...|...|

## Performances

The results of different experiments are shown in the table below. The mIoU values shows that overall using intensity and color values is useful when distinguishing between pole attachments. Particularly, the improvement reflects in distinguishing different types of signs better as they have high reflectivity and distinct colors. You can also look at two examples in the media folder to gain insight about the qualitative results. 

| Method | KPConv (default) | KPConv (debugged) | KPConv (x,y,z,i) |  KPConv (x,y,z,r,g,b) | KPConv (x,y,z,r,g,b,i) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| mIoU | 36% |  84.4%  |  84.9%  |  84.8%  | 87.5% |


## Acknowledgements

Our code is based on KPConv algorithm [KPConv](https://github.com/HuguesTHOMAS/KPConv). You can read more about the algorithm [here](https://arxiv.org/abs/1904.08889) <br />
Our data provider is [Cyclomedia](https://www.cyclomedia.com/nl). <br />
To track the experiments [wandb](https://wandb.ai/home) dashboard is used. <br />



