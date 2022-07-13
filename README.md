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

#### If you are running the pipeline for a customm dataset:  <br />

* Before running the training algorithm, split your data using the stratified split code (stratified_split) This code is reading laz format. Note that for handling .ply files requires some modifications.
* Prepare the data using Prepare_data_xyzrgbi.py . This code prepares .laz files, saves them as .ply with normalized coordinates, color and intensity values and gives them proper names. 
* Create a Data folder inside the KPConv folder and add the three folders of 'train_ply', 'val_ply', and 'test_ply' containing the prepared point clouds to it
* Modify the 'self.path' variable in the ShapenNetPart.py in the ./KPConv/datasets folder to read the point clouds from the Data folder
* Determine whether the intensity and color values are going to be fed into the model in the training_ShapeNetPart (color_info, intensity_info) and in ./KPConv/utils/config.py.
* Run the training_ShapeNetPart.py

#### If you are runnning on our provided dataset: <br />

* Copy the folders inside the 'data' folder to ./KPConv/Data
* Modify the 'self.path' variable in the ShapenNetPart.py in the ./KPConv/datasets folder to read the point clouds from the Data folder
* Determine whether the intensity and color values are going to be fed into the model in the training_ShapeNetPart (color_info, intensity_info) and in ./KPConv/utils/config.py.
* Run the training_ShapeNetPart.py

## Performances

The following tables report the current performances on different tasks and datasets. 

### Part Segmentation of Pole-like objects 

| Method | KPConv (default) | KPConv (debugged) | KPConv (x,y,z,i) |  KPConv (x,y,z,r,g,b) | KPConv (x,y,z,r,g,b,i) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KPConv _deformable_      | 36% |  84.4%  |  84.9%  |  84.8%  | 87.5% 


