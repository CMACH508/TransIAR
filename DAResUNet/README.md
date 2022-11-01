# DAResUNet



DAResUNet's pytorch modified version: from segmentation to classification

> A clinically applicable deep-learning model for detecting intracranial aneurysm in computed tomography angiography images
>
> Paper URL: https://www.nature.com/articles/s41467-020-19527-w



## Introduction

DAResUNet is a 3-dimensional (3D) CNN for the segmentation of IAs from digital subtraction bone-removal CTA images to evaluate the presence and locations of aneurysms. We modified it into a classification model (encoder+classifier, unet+classifier) for aneurysm rupture risk prediction, to compare with our method.



## Overview

<img src=".\overview.jpg" width="100%" />



## Train and test

To train the encoder+classifier, run

```
python train --is_unet 1
```

To train the unet+classifier (with reconstruction loss), run

```
python train --is_unet 0
```

To test the encoder+classifier on Balanced Dataset, run

```
python test_enc.py
```

To test the unet+classifier on Balanced Dataset, run

```
python test_unet.py
```



Note that only `python test_enc.py` and `python test_unet.py` can be executed successfully due to data inaccessibility, while you can train and test the method on your own datasets.

To run `python test_enc.py` and `python test_unet.py`, download `model_enc.pth` (https://drive.google.com/file/d/180TT8ko24eBsFIucsuFvSDIQiEpj7kBR/view?usp=sharing) and `model_unet.pth` (https://drive.google.com/file/d/19CKXwRnu--zYtJRYWBoln8Sg_6ZOhI8u/view?usp=sharing) to ./checkpoint.



## Pre-trained Models

`model_enc.pth` is the trained model of encoder+classifier, whose accuracy on Balanced Dataset is 84.15.

`model_unet.pth` is the trained model of unet+classifier, whose accuracy on Balanced Dataset is 82.93.

