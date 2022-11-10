# TransIAR-Net

This repository contains the source code, trained model and part of the test set for our work: Towards an end-to-end prediction of rupture status of intracranial aneurysm via deep learning. 



## Introduction

Intracranial aneurysms (IA) seriously threaten human health, and ruptured IA may even cause death of patients. Early assessment on the risk of IA rupture is important for timely and appropriate treatment for patients. To address this issue, we propose an end-to-end deep learning model (TransIAR net) for accurate IA rupture status prediction, without laborious human efforts on hand-crafted morphological features. It takes 3D computed tomography angiography (CTA) data as input and automatically extracts features of IA geometry and neighborhood information. Quantitative experiments demonstrate the superior performance of the proposed method. To the best of our knowledge, TransIAR net is the first end-to-end method for IA rupture prediction directly on the 3D CTA data. It is a promising artificial intelligence tool to assist the doctors for efficient, accurate clinical diagnosis and timely treatment of IA patients.



## Overview

<img src=".\overview.png" width="100%" />



## Requirements

python==3.8

einops==0.5.0

numpy==1.22.3

scikit_learn==1.1.3

torch==1.12.1



Install dependencies:

```
pip install -r requirements.txt
```



## Dataset

`dataset_cta_balanced_train.pkl` and `dataset_cta_balanced_test.pkl` are 3D CTA cubes of Balanced Dataset, each case augmented 32 times.

`dataset_af_balanced_train.pkl` and `dataset_af_balanced_test.pkl`  are the corresponding auxiliary features of Balanced Dataset.



We only provide the balanced test set (82 cases) [`dataset_cta_balanced_test.pkl`](https://drive.google.com/file/d/100Pa_vtNoRGIlk5Q0RruWFVj5WtcN8H-/view?usp=sharing) and [`dataset_af_balanced_test.pkl`](https://drive.google.com/file/d/1HYA-EAzCp8D5m1xYQpqnNn__63Nya05e/view?usp=sharing) due to hospital regulation restrictions and patient privacy concerns. 



To test our model, download `dataset_cta_balanced_test.pkl` and `dataset_af_balanced_test.pkl` to /dataset folder and then run `python test_wo_af.py` and `python test_w_af.py`, which are expected to get the following output respectively,

```
Accuracy: 89.02
Precision: 88.1
Recall: 90.24
F1 score: 89.16
AUC: 92.15
AUPR: 93.16
```

```
Accuracy: 91.46
Precision: 94.74
Recall: 87.8
F1 score: 91.14
AUC: 92.09
AUPR: 89.58
```



## Train and test

`models.py` contains two models we proposed, one is TransIAR (TransIAR-Net without auxiliary features), the other is TransIAR_AF (TransIAR-Net with auxiliary features), i.e. the final model.

`transformer.py` contains the transformer module.

`utils.py` contains some functions for calculating metrics.



To train TransIAR, run

```
python train_wo_af.py
```

To train TransIAR_AF , run

```
python train_w_af.py
```



To test TransIAR on Balanced Dataset, run

```
python test_wo_af.py
```

To test TransIAR_AF on Balanced Dataset, run

```
python test_w_af.py
```



Note that only `python test_wo_af.py` and `python test_w_af.py` can be executed successfully due to data inaccessibility, while you can train and test our method on your own datasets.



## Pre-trained Models

`model_wo_af.pth` is the trained model of TransIAR, whose accuracy on Balanced Dataset  is **89.02**.

`model_w_af.pth` is the trained model of TransIAR_AF, whose accuracy on Balanced Dataset  is **91.46**.



## Comparisons

We consider relevant SOTA 2D or 3D models in the literature, including the work of Kim et al., M3T and DAResUNET, and reproduce them for comparisons. It should be noted that the existing 3D models are not for the IA rupture status prediction problem, and thus we modify them appropriately to adapt to the problem setting in this paper.

In addition to the Balanced and Imbalanced datasets, we also collected CTA images of 43 patients from other four hospitals and merge them as an external test set. 

We evaluate our model and other competing methods on the external test set without auxiliary features. The results are as follows:

| Models         | Accuracy  | Precision | Recall    | AUC       | AUPR      | F1 score  |
| -------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| $RF^m$         | 65.12     | 76.92     | 68.97     | 72.66     | 83.99     | 72.73     |
| $SVM^m$        | 58.14     | 82.35     | 48.28     | 80.05     | 89.79     | 60.87     |
| $Kim\ et\ al.$ | 65.12     | **100.0** | 48.28     | 91.63     | 96.01     | 65.12     |
| $M3T$          | 76.74     | 77.14     | **93.10** | 82.76     | 89.38     | 84.38     |
| $DAResUNET^b$  | 79.07     | 95.45     | 72.41     | 95.32     | 97.70     | 82.35     |
| $DAResUNET^c$  | 72.09     | 86.96     | 68.97     | 89.66     | 94.27     | 76.92     |
| $IAR$          | 90.70     | 96.30     | 89.66     | 95.32     | 97.80     | 92.86     |
| $TransIAR$     | **93.02** | **100.0** | 89.66     | **98.03** | **99.17** | **94.55** |

The results on the external test set shows that TransIAR has good generalization performance and is potentially a useful model for clinical utility to help radiologists in the assessment of IA rupture risk.
