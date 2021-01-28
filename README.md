# Investigate Wasserstein Auto-Encoder for Unsupervised Anomaly Detection in Brain MRI

This repository contains code for my Master Research Project (P2).

## Folder Structure
Below is the folder structure for this repository:
```
UAD_brain_MRI.ipynb 
```
* python integrated notebook that contain all master code. This file can be run on local machine (Jupyter Notebook) and also cloud platform (Google Collaboratory).
```
run.py
```
**python code for user defined function on reset default graph, handle additional Config parameters, create an instance of the model and train it, evaluate best dice, evaluate generalization
```
run.py
```
* requirements.txt -> list of python packages and its version

Folder Hierarchy level:
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
  ├── mains/ - Main files to train each architecture
  │   ├── main_AE.py
  │   └── ...
  │
  ├── model/ - Architecture definitions
  │   ├── autoencoder.py
  │   └── ...
  │
  ├── trainers/ - trainers including definition of loss functions, metrics and restoration methods
  │   ├── AE.py
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
```

## Usage
Data consumed in this project can be obtained from Brainweb website. The modality parameters can be custom and controlled by user. the dataset can be downloaded [here](https://brainweb.bic.mni.mcgill.ca/).



