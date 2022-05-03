# Pole-Part-Segmentation

Created by Sara Yousefimashhoor as the extension of Hugues THOMAS Deep learning Algorithm (KPConv)

## Introduction


This repository contains the implementation of **Kernel Point Convolution** (KPConv), a point convolution operator 
to decompose pole-like objects to its constituent parts (carrier and different attachments). 


![Intro figure](https://github.com/Amsterdam-Internships/Pole-Part-Segmentation/blob/master/Picture1.png)


## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). 


## Achieved Milestones

•	Feeding the municipality's custom dataset to a publicly available deep learning algorithm (KPConv) and introducing a new 'pole' class to the model. 
•	Debugging the KPConv codes for the part-segmentation task.
•	Including additional features in the training and testing pipeline(RGB and intensity values)
•	Fine-tuning a selection of parameters 
•	Improving the model performance

 

## Performances

The following tables report the current performances on different tasks and datasets. 

### Part Segmentation of Pole-like objects 

| Method | KPConv (default) | KPConv (debugged) | KPConv (x,y,z,i) |  KPConv (x,y,z,r,g,b) | KPConv (x,y,z,r,g,b,i) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| KPConv _rigid_      | 36% |  84.4%  |  84.9%  |  84.8%  | 87.5% 


