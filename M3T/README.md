# M3T

PyTorch implementation of M3T

> M3T: three-dimensional Medical image classifier using Multi-plane and Multi-slice Transformer
>
> Paper URL : https://openaccess.thecvf.com/content/CVPR2022/html/Jang_M3T_Three-Dimensional_Medical_Image_Classifier_Using_Multi-Plane_and_Multi-Slice_Transformer_CVPR_2022_paper.html



## Introduction

M3T is a three-dimensional Medical image classifier using Multi-plane and Multi-slice Transformer (M3T) network to classify Alzheimer’s disease (AD) in 3D MRI images. It synergically combines 3D CNN, 2D CNN, and Transformer for accurate AD classification. 

Since the authors didn't provide the source code, we implement the method as closely as possible according to the paper, slightly modify it to take CTA images as input, and train and test on our dataset.



## Overview

<img src=".\overview.jpg" width="100%" />



## Train and test

To train M3T, run

```
python train.py
```

To test M3T on Balanced Dataset, run

```
python test.py
```



Note that only `python test.py` can be executed successfully due to data inaccessibility, while you can train and test the method on your own datasets.

To run `python test.py`, download `model.pth` (https://drive.google.com/file/d/1Nm-2av5Dn42PCmeOpwHMnxJURvVOMj2Z/view?usp=share_link)  to ./checkpoint.



Note that the input above is 96-sized patches (without BFS). For a fair comparison, we use 48+96 (without BFS) as the input:

To train M3T, run

```
python train_2c.py
```

To test M3T on Balanced Dataset, run

```
python test_balanced_2c.py
```

To run `python test_balanced_2c.py`, download `model_0.8537.pth` (https://drive.google.com/file/d/1WaiuffEZ8Hby-ohX4CTi7oZcvFk9mOp4/view?usp=share_link)  to ./checkpoint.



## Pre-trained Model

`model.pth` is the trained model of M3T, whose accuracy on Balanced Dataset is **80.49**.

`model_0.8537` is the trained model of M3T with 2 channel input, whose accuracy on Balanced Dataset is **85.37**.