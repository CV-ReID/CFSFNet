# Introduction
This is the source code of our CFSFNet paper "Channel-level Feature Selection and Fusion Network for Visible-infrared Person Re-identification". Please cite the following paper if you use our code.


# Dependencies
* Python 3.7.11
* PyTorch 1.10.1
* cudatoolkit 11.3.58
* torchvision 0.11.2

# Data Preparation
Download the [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) dataset and the [RegDB](http://dm.dongguk.edu/link.html) dataset, and place them to /home/seayuan/Dataset/ folders.

# Usage
* Start training by executing the following commands.

1.For SYSU-MM01 dataset:

Train: ```
python train.py --dataset sysu  --gpu 1``` 

Test: ```
python test.py --dataset sysu --model_path 'path' --gpu 1``` 

2.For RegDB dataset:

```python train.py --dataset regdb  --gpu 1```


For any questions, feel free to contact us.
