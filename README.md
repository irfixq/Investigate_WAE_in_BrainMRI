# Investigate Wasserstein Auto-Encoder for Unsupervised Anomaly Detection in Brain MRI

This repository contains code for my Master Research Project (P2).

**Disclaimer:
The code has been cleaned and polished for the sake of clarity and reproducibility, and even though it has been checked thoroughly, it might contain bugs or mistakes. Please do not hesitate to open an issue or contact the authors to inform of any problem you may find within this repository. 


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
Utilities to simplify working with TFRecord files and TensorFlow data pipeline. This is a simple format for storing a sequence binary records for efficient serialization of structured data. each observation values need to be converted to a [tf.train.Feature](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample) by creating dataset using NumPy.

8. utils.py - 
Configuration file to visualize the data using python and export back as .pdf file.

#### Folder: trainers
Trainers including definition of loss functions, metrics and restoration methods. Contain 'DLMODEL.py' and 'AEMODEL.py' as baseline class for all Deep Learning needs with TensorFlow. All trainers code also contain early stopping as the validation method. the code is written in separate python file for each trainer.

#### Folder: models
Contain model architecture definitions.


#### Folder: mains
Main files to train each architecture.


#### Folder: logs
Just create an empty folder to store tensorboard logs.


#### Folder: dataloaders
Contain user defined functions to read Brainweb data. More information about Brainweb data format in NII can be retrieved [here](https://radiopaedia.org/articles/nifti-file-format).


#### Folder: Brainweb
Folder to store your downloaded dataset from Brainweb website. Make sure to configure the path properly in config.default.json file. In this folder, i put sample of data in .mnc.gz format , pckl file and tfrecord file.


### Folder Hierarchy level:
```
  Unsupervised_Anomaly_Detection_Brain_MRI/
  │
  ├── UAD_brain_MRI.ipynb 
  ├── run.py 
  ├── config.default.json
  ├── requirements.txt 
  ├── GPU_configuration.txt 
  │
  ├── data_loaders/ 
  │   ├── BRAINWEB.py
  │
  ├── logs/  
  │
  ├── mains/ 
  │   ├── main_AE.py
  │   ├── main_CE.py
  │   ├── main_ceVAE.py
  │   ├── main_fAnoGAN.py
  |   ├── main_GMVAE.py
  │   ├── main_VAE.py
  │   └── main_WAEGAN.py
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
  │   ├── VAE.py
  │   ├── CE.py
  │   ├── ceVAE.py
  │   ├── GMVAE.py
  |   ├── fAnoGAN.py
  │   ├── anoVAEGAN.py
  |   ├── WAEGAN.py
  |   ├── Metrics.py
  │   ├── trainer_utils.py
  │   ├── AEMODEL.py
  │   └── DLMODEL.py
  │  
  └── utils/ 
  │   ├── default_config_setup.py
  |   ├── Evaluation.py
  │   ├── image_utils.py
  |   ├── logger.py
  |   ├── MINC.py
  │   ├── NII.py
  │   ├── tfrecord_utils.py
  │   └── utils.py

```

## Usage
Data consumed in this project can be obtained from Brainweb website. The modality parameters can be custom and controlled by user. the dataset can be downloaded [here](https://brainweb.bic.mni.mcgill.ca/).

**Disclaimer: 
The data was collected on July - October 2020. Hence, you might need to tune some hyperparameter and dataloader to fit into the latest Brainweb dataset as the website might update their data and format.



