# Deep Learning/Machine Learning Application in Financial prediction

## Introduction

This Project is a experiment of applying Kernel Tricks, Neural Network with sparse coding and RNN methods in predicting financial data. The first two 

methods are applied to cross-sectional data, the third method is used to predicting time series data. 

## Code overview

### AssetPricing

The AssetPricing file includes modules specifically designed for practical applications in asset pricing. These modules are tailored to facilitate efficient handling and analysis of real-world financial data.

#### _auto_pca.py

This file contains the neural network class (MLP with sparse coding) and dataloader that will be used for cross sectional prediction task. Moreover, the custom loss funtion "negative_correlation_loss" is also included. 

#### _kernel_methods.py 

This file contains the kernel class that will be used to predict cross sectional data.

#### custom_rnn_scratch.py

This file contains the self-designed class of RNN network, it will allow the updatation process of hidden states have more complexity and more layers to go through.

#### gru_scratch.py

This file contain the self-designed Gated Recurrent Unit which embedding the rnn cell in the file "custom_rnn_scratch.py", also the gradient clipping function "grad_clipping()" is in it. 

### _auto_pac.ipynb

This file is the notebook using MLP to predict the cross sectional data

### testing_for_kernel.ipynb

This file is the notebook using kernel Methods to predict cross sectional data

### Custom_GRU.ipynb

This file is the notebook using RNN model to predict time series data





