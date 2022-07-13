### Installation instructions for Ubuntu 20.04
     
* Make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configurations have been tested: 
     - TensorFlow-gpu 1.12.0, CUDA 9.0 and cuDNN 7.6.5 <br />
  <br />
  It is recommended to create an environment using conda with the mentioned configuration.<br />
  If you are using CUDA 11.3, then use TensorFlow-gpu 1.13.0 instead.<br />

* After installing Anaconda:

          conda create -n tf-gpu cudatoolkit==9.0 tensorflow-gpu==1.12.0
          conda activate tf-gpu

* Install the other dependencies with conda:
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

* Change the `-D_GLIBCXX_USE_CXX11_ABI=0` flag for each line in `tf_custom_ops/compile_op.sh` to `-D_GLIBCXX_USE_CXX11_ABI=1`

Now you are ready to run any code in KPConv folder.

### Packages Versions Used

| Package | cudatoolkit | cudnn | tensorflow-gpu | Python | NumPy | scikit-learn | PyQt | psutil | 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Version | 9.0 | 7.6.5 | 1.12.0 | 3.6.13 | 1.19.2 | 0.24.2 | 5.9.2 | 5.9.0 |
