# Kim_2D



PyTorch implementation of Kim's Rupture Risk Assessment Method

> Machine Learning Application for Rupture Risk Assessment in Small-Sized Intracranial Aneurysm
>
> Paper URL : https://www.mdpi.com/2077-0383/8/5/683



## Introduction

This method utilizes a 2D CNN network to extract features from 2D IA images to assess the risk. A retrospective data set, including 368 patients, was used as a training cohort for the CNN. Aneurysm images in six directions were obtained from each patient and the region-of-interest in each image was extracted. The resulting CNN was prospectively tested in 272 patients and the sensitivity, specificity, overall accuracy, and receiver operating characteristics (ROC) were compared to a human evaluator.

Since the authors didn't provide the source code, we implemented the method as closely as possible according to the paper, and train and test on our dataset. We project aneurysm region to 2D images from six directions, which are used as model's input.



## Overview

<img src=".\overview.jpg" width="100%" />



## Train and test

To train the model using projection data, run

```
python train.py
```

To test the model on Balanced Dataset of projection data, run

```
python test.py
```

To test the model on Imbalanced Dataset of projection data, run

```
python test_imbalanced.py
```



Note that only `python test.py`  can be executed successfully due to data inaccessibility, while you can train and test the method on your own datasets.



To run `python test.py`, download `dataset_2d_test.pkl` (https://drive.google.com/file/d/1aUdRKDiWhpFkGlz_l6Op6QvaAaqZ8S7I/view?usp=sharing) and to ../../dataset_2d, download `model.pth` (https://drive.google.com/file/d/1sp526Oz5GovLXZXU-RD6xEETIqm4-mHi/view?usp=sharing) to ./checkpoint.



## Pre-trained Models

`model.pth` is the trained model on projection data, whose accuracy on Balanced Dataset is 79.27.
