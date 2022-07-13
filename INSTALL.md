### Installation instructions for Ubuntu 20.04
     
* Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configurations have been tested: 
     - TensorFlow-gpu 1.12.0, CUDA 9.0 and cuDNN 7.6.5 <\b >
  It is recommended to create an environment using conda with the mentioned configuration.<\b >
  If you are using CUDA 11.3, then use TensorFlow-gpu 1.13.0 instead.<\b >
  
* Ensure all python packages are installed :

          sudo apt update
          sudo apt install python3-dev python3-pip python3-tk

* Follow <a href="https://www.tensorflow.org/install/pip">Tensorflow installation procedure</a>.

* Install the other dependencies with pip:
     - numpy
     - scikit-learn
     - psutil
     - matplotlib (for visualization)
     - mayavi (for visualization)
     - PyQt5 (for visualization)
     - wandb (for tracking the process)
     
* Compile the customized Tensorflow operators located in `tf_custom_ops`. Open a terminal in this folder, and run:

          sh compile_op.sh

     N.B. If you installed Tensorflow in a virtual environment, it needs to be activated when running these scripts
     
* Compile the C++ extension module for python located in `cpp_wrappers`. Open a terminal in this folder, and run:

          sh compile_wrappers.sh

You should now be able to train Kernel-Point Convolution models

### Additional Step for Ubuntu 18.04 and 20.04 (Thank to @noahtren)

* Change the `-D_GLIBCXX_USE_CXX11_ABI=0` flag for each line in `tf_custom_ops/compile_op.sh` to '-D_GLIBCXX_USE_CXX11_ABI=1'



* Before running the training algorithm, split your data using the stratified split code (stratified_split) This code is reading laz format. for handling .ply files requires some modifications
* Prepare the data using all_data_xyzrgbi.py . This code prepares .laz files, save them as .ply with coordinates, color and intensity  normalized values. 
* Rname the files using Rename_Files.py -
* Create a Data folder in the KPConv folder and add the three folders of 'train_ply', 'val_ply', and 'test_ply' to it
* Modify the ShapenNetPart.p in the dataset folder where it is defining the path to the dataset 
* Determine whether the intensity and color values are going to be fed into the model in the training_ShapeNetPart (color_info, intensity_info)
* Run the training_ShapeNetPart.py

# ATTENTION: In all the files you should give the custom path to your modified data in the previous step. 
