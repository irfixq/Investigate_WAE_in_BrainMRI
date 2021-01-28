# Investigate Wasserstein Auto-Encoder for Unsupervised Anomaly Detection in Brain MRI

This repository contains code for my Master Research Project (P2).

## Folder Structure
Below is the file description for this repository:

#### UAD_brain_MRI.ipynb 
* Python integrated notebook that contain all master code. This file can be run on local machine (Jupyter Notebook) and also cloud platform (Google Collaboratory).


#### run.py
* Python code for user defined function on reset default graph, handle additional Config parameters, create an instance of the model and train it, evaluate best dice, evaluate generalization.


#### requirements.txt
* List of python packages and its version.


#### GPU_configuration.txt
* Steps to configure GPU using CUDA core on local machine or cloud platform. This research is using TensorFlow with GPU support. More information about GPU installation can be retrieved [here](https://www.tensorflow.org/install/gpu) and some guide for GPU usage [here](https://www.tensorflow.org/guide/gpu).


#### config.default.json
* Configuration file for path directory. Below is the step to follow to configure this JSON file:
1. Define source path of dataset
2. Update dataloaders if you want to use your own dataset other than Brainweb dataset. However, Brainweb could update their database from time to time. Hence, few code enhancement is needed especially in dataloaders


#### Folder: utils
Contain small utility functions written in python such as for;
1. Evaluation.py - 
For image reconstruction, confusion matrix calculation, logistic function to squash reconstruction error, expand dimension, kernel size, compute detection rate for predicted volume and ground truth volume, determine number of training samples, iteration over all unhealthy data, sanity checks, get sample data without dropout, data normalization, evaluate unhealthy samples (lesion), compute ROC curev and PRC curve.

2. MINC.py - 
Contain python package ([NiBabel](https://nipy.org/nibabel/)) for read and write access to neuroimaging file format which in this case is to convert MINC format to NII/Nifti format.

3. NII.py - 
Contain code for evaluating segmentation results using [SimpleITK](https://simpleitk.org/) package. This code also used to visualize the NII data view mapping.

4. default_config_setup.py - 
Contain python code to setup the class and user defined function from Brainweb dataset.

5. image_utils.py - 
Contain image dimension configuration and user defined function for prediction and groundtruth to image.

6. logger.py - 
Contain public TensorFlow interface to summarize training, validation and testing phase. More detail about each variable used in this code can be found [here](https://www.tensorflow.org/api_docs/python/tf/compat/v1#functions).

7. tfrecord_utils.py - 


8. utils.py - 


#### Folder: trainers
Trainers including definition of loss functions, metrics and restoration methods.


#### Folder: models
Contain model architecture definitions.


#### Folder: mains
Main files to train each architecture.


#### Folder: logs
Just create an empty folder to store tensorboard logs.


#### Folder: dataloaders
Contain user defined functions to read Brainweb data. More information about Brainweb data format in NII can be retrieved [here](https://radiopaedia.org/articles/nifti-file-format).


#### Folder: Brainweb
Folder to store your downloaded dataset from Brainweb website. Make sure 


### Folder Hierarchy level:
```
  Unsupervised_Anomaly_Detection_Brain_MRI/
  │
  ├── Unsupervised Anomaly Detection Brain-MRI.ipynb - Jupyter notebook to work on Google Colab
  ├── run.py - execute to run in commandline
  ├── config.json - holds configuration
  │
  ├── data_loaders/ - Definition of dataloaders
  │   ├── BRAINWEB.py
  │
  ├── logs/ - default directory for storing tensorboard logs 
  │
  ├── mains/ 
  │   ├── main_AE.py
  │
  ├── model/ 
  │   ├── autoencoder.py
  │   ├── variational_autoencoder.py
  │   ├── context_encoding_autoencoder.py
  │   ├── context_encoding_variational_autoencoder.py
  │   ├── Gaussian_mixture_variational_autoencoder.py
  |   ├── fAnoGAN.py
  │   ├── anoVAEGAN.py
  │   └── WAEGAN.py
  │ 
  ├── trainers/ 
  │   ├── AE.py
  │   └── 
  │  
  └── utils/ 
      ├── util.py
      └── 
```

## Usage
Data consumed in this project can be obtained from Brainweb website. The modality parameters can be custom and controlled by user. the dataset can be downloaded [here](https://brainweb.bic.mni.mcgill.ca/).



