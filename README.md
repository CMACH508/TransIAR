# TransIAR-Net



## requirements

python 3.8

einops 0.4.1

scikit_learn 1.1.2

torch 1.7.1+cu101



## dataset

`dataset_cta_balanced.pkl` is generated from Balanced Dataset, each case augmented 32 times.

`dataset_cta_noused.pkl` contains  unused 167 cases of ruptured IAs, each case augmented 32 times.

`dataset_af_balanced.pkl` is the corresponding auxiliary features of dataset_cta_balanced.pkl.

`dataset_af_noused.pkl` is the corresponding auxiliary features of dataset_cta_noused.pkl.



## src

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



## checkpoint

`model_wo_af.pth` is the trained model of TransIAR, whose accuracy on Balanced Dataset  is **89.02**.

`model_w_af.pth` is the trained model of TransIAR_AF, whose accuracy on Balanced Dataset  is **91.46**.