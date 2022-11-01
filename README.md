# TransIAR-Net

It is critical to accurately predict the rupture risk of an intracranial aneurysm (IA) for timely and appropriate treatments, because the fatality rate after rupture is 50%. The existing machine learning methods require neuroradiologists to manually measure the characteristics of IA's morphology to predict the risk of aneurysm rupture, which is time-consuming and labor-intensive and barely considers the IA neighborhood information. In this paper, we propose an end-to-end deep learning method, i.e., TransIAR net, for the task of aneurysm rupture prediction. A multi-scale deep 3D CNN is developed to automatically extract the morphological features of IA and its neighborhood information directly from the raw 3D CTA image data. A transformer module is devised to model the spatial dependence within the 3D CNN embeddings of the aneurysm and its surrounding anatomical structures, and the representation learning is strengthened to be more discriminative and predictive for the rupture prediction.
We evaluate the TransIAR net by experiments on both balanced and unbalanced datasets. The prediction performance becomes much better when replacing the hand-crafted features by the neuroradiologists with the features learned by the TransIAR net in a traditional machine learning model like RF or SVM. The performance is further improved when the representation learning and classifier construction are jointly optimized by the TransIAR net. To the best of our knowledge, the TransIAR net is the first end-to-end model with fast and accurate prediction on the IA rupture risk from the raw 3D CTA data, which is a promising tool to assist the doctors in clinical practice.



## Overview

<img src=".\overview.png" width="100%" />



## Requirements

python==3.8

einops==0.4.1

scikit_learn==1.1.2

torch==1.7.1+cu101



## Dataset

`dataset_cta_balanced_train.pkl` and `dataset_cta_balanced_test.pkl` are 3D CTA cubes generated from Balanced Dataset, each case augmented 32 times.

`dataset_cta_noused.pkl` contains 3D CTA cubes of unused 167 cases (ruptured IAs), each case augmented 32 times.

`dataset_af_balanced_train.pkl` and `dataset_af_balanced_test.pkl`  are the corresponding auxiliary features of Balanced Dataset.

`dataset_af_noused.pkl` is the corresponding auxiliary features of `dataset_cta_noused.pkl`.



We only provide the preprocessed balanced test set (82 cases)  `dataset_cta_balanced_test.pkl` (https://drive.google.com/file/d/100Pa_vtNoRGIlk5Q0RruWFVj5WtcN8H-/view?usp=sharing) and `dataset_af_balanced_test.pkl` (https://drive.google.com/file/d/1HYA-EAzCp8D5m1xYQpqnNn__63Nya05e/view?usp=sharing) due to hospital regulation restrictions and patient privacy concerns. 



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

To test TransIAR on Imbalanced Dataset, run

```
python test_wo_af_imbalanced.py
```



To test TransIAR_AF on Balanced Dataset, run

```
python test_w_af.py
```

To test TransIAR_AF on Imbalanced Dataset, run

```
python test_w_af_imbalanced.py
```



Note that only `python test_wo_af.py` and `python test_w_af.py` can be executed successfully due to data inaccessibility, while you can train and test our method on your own datasets.



## Pre-trained Models

`model_wo_af.pth` is the trained model of TransIAR, whose accuracy on Balanced Dataset  is **89.02**.

`model_w_af.pth` is the trained model of TransIAR_AF, whose accuracy on Balanced Dataset  is **91.46**.



