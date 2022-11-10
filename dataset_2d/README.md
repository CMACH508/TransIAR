The following dataset are used by the 2D method (Kim_2D):
`dataset_2d_test.pkl` (https://drive.google.com/file/d/1aUdRKDiWhpFkGlz_l6Op6QvaAaqZ8S7I/view?usp=sharing) 



`dataset_2d_test.pkl` contains an array of shape $[492, 3, 224, 224]$ (2D CTA cube projections) and an array of shape $[492]$ (labels), where 

$492 = 82 \times 6$ ($82$ means that the balanced test set is consist of 82 cases, and $6$ means the aneurysm region is projected to 2D images from six directions),

$3$ is the number of input channels,

$[224,224]$ is the shape of 2D images.