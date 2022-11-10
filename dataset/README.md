The following dataset are used by TransIAR and other 3D methods:
`dataset_cta_balanced_test.pkl` (https://drive.google.com/file/d/100Pa_vtNoRGIlk5Q0RruWFVj5WtcN8H-/view?usp=sharing)
`dataset_af_balanced_test.pkl` (https://drive.google.com/file/d/1HYA-EAzCp8D5m1xYQpqnNn__63Nya05e/view?usp=sharing)



`dataset_cta_balanced_test.pkl` contains an array of shape $[2624, 4, 48, 48, 48]$ (CTA cubes) and an array of shape $[2624]$ (labels), where 

$2624 = 82 \times 32$ ($82$ means that the balanced test set is consist of 82 cases, and $32$ means that each case is rotated 32 times),

$4$ is the number of input channels,

$[48,48,48]$ is the shape of 3D CTA cubes.

`dataset_af_balanced_test.pkl` contains an array of shape $[2624, 5]$, where $5$ is the number of auxiliary features.

