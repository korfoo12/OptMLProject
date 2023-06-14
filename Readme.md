# Optimization methods for Federated Learning

### Team members

* Tran Huong Lan (lan.tranhuong@epfl.ch)
* Larigauderie Léo Benjamin Paul Killian (leo.larigauderie@epfl.ch) 
* Ivan Korovkin (ivan.korovkin@epfl.ch)

### Description

This github repository contains the results for the mini-project of "Optimization for Machine learning" (CS-439) course at EPFL. The goal was to gather insights on the performance of different first-order optimizers within Federated Learning setting. In our study, we used CIFAR-10 dataset and split it between 8 clients. Moreover, we studied both algorithm performances for iid and non-iid data split.


### Required libraries

The project was done using Python 3.8.9 with the following libraries:
* torch
* torchvision
* more-itertools
* random
* matplotlib
* numpy
* sklearn

### Repository structure

Our repository has the following structure:
```
.
├── code                                    # Directory with code implementation
│   ├── lib                                 # Directory with helper .py files
│   │   ├── client.py                       # Cient class implementation
│   │   ├── data_helper.py                  # Dataset helper functions
│   │   ├── models.py                       # Model used in training
│   │   ├── plots.py                        # Plots for training metrics
│   │   ├── server.py                       # Server class implementation
│   │   └── train_helper.py                 # Helper functions for training
│   ├── CV_Adagrad_non_iid.ipynb            # Cross validation(CV) results for Adagrad learning rate (lr) on non-iid data
│   ├── CV_Adam.ipynb                       # CV results for Adam optimizer lr and betas, both non-iid and iid
│   ├── CV_SGD_ADAGRAD_iid.ipynb            # CV results for SGD and Adagrad lr on iid data
│   ├── CV_SGD_non_iid.ipynb                # CV results for SGD optimizer lr on non-iid data
│   ├── CV_momentum_lr_iid.ipynb            # CV results for momentum SGD lr on iid data
│   ├── CV_momentum_lr_non_iid.ipynb        # CV results for momentum SGD lr on non-iid data
│   ├── CV_momentum_params.ipynb            # CV results for momentum SGD momentum parameter on iid and non-iid data
│   └── run.ipynb                           # Main file to run to get the results
├── Report.pdf                              # Report for the project
└── README.md
```

### Reproducing the results


Thank you for paying attention to our project!

LIT team
