# AME_official

The official source code for the paper "Improving Adversarial Robustness of Deep Neural Networks via Adaptive Margin Evolution" 

This paper is published in Neurocomputing (https://doi.org/10.1016/j.neucom.2023.126524)

# Requirements

Python3.8.10

Pytorch1.9.0

advertorch (https://github.com/BorealisAI/advertorch)

autoattack (https://github.com/fra31/auto-attack)


# Guide
### For CIFAR10

Run CIFAR10_CNNM_AME.py to get the proposed AME model

Run CIFAR10_CNNM_ce.py to get the clean model

### For SVHN

Run SVHN_CNN_AME.py to get the proposed AME model

Run SVHN_CNN_ce.py to get the clean model

### For Tiny ImageNet

Run TIM_CNN_AME.py to get the proposed AME model

Run TIM_CNN_ce.py to get the clean model

## The trained models

The trained models are available at https://drive.google.com/drive/folders/1B-hjpdQ45_xQtqTVc0nOEV4DA1FylFcy?usp=sharing

The random seed is 1. 

# Contact

Should you have any questions, please feel free to contact liang.liang@miami.edu or l.ma@miami.edu
