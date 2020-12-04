# STEM: An approach to Multi-source Domain Adaptation with Guarantees

<p align="center">
  <img src="images/overal_framework.png" /> 
</p>

## Introduction
This is the implementation of ``STEM: An approach to Multi-source Domain Adaptation with Guarantees''

### Prerequisites
System Requirement:
* Ubuntu 16.04
* Anaconda3
* Cuda toolkit 10.0

Install other environment requirement by Anaconda3 following:
```
conda env create -f env.yml
```
### Dataset Preparation
Please download and unzip the dataset and save under `../datasets`. To save time computing, we extracted ResNet101 feature for Office-Caltech10 and provided as following:
* [DomainNet](http://ai.bu.edu/M3SDA/)

### SOTA mode
`4_sep_1_2_confi_1` 
Should set mode to this to get SOTA result

### Training
Run:  
`python main_multi_g.py --config DomainNet.yaml --target-domain clipart -bp base_path -j 8 -t 1 --gpu 0,1,2 -mode 4_sep_1_2_confi_1 -bz 200 -te 1000`

`--config` config file for training  
`--target-domain` target name: clipart, infograph, painting, quickdraw, real, sketch  
`-bp` base path to save output (log file, checkpoint)  
`-j` job of number of worker. depend on your system, should set to 8 to get fastest speed  
`-t` time training the same target domain and same mode. (just using to create name of log file)  
`--gpu` gpu id to train model, batch-size should be set depend on GPU capability  
`-mode` important!!! using different mode will change the training process and change result  
`-bz` batch-size  
`-te` total epoch to train  

Note that the best model will be save during training