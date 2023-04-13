import numpy as np
from utils import read_nii, save_nii, crop_v, process
from scipy.ndimage.interpolation import zoom


thresh = 0.6
idx = '0000'
path = './CTA/{}.nii.gz'.format(idx)

cropped_vox_large = crop_v(idx, 96)
cropped_vox_small = crop_v(idx, 48)
data_large_p = process(cropped_vox_large, thresh)
data_small_p = process(cropped_vox_small, thresh)
cropped_vox_large = zoom(cropped_vox_large, zoom = 0.5, order=1)
data_large_p = zoom(data_large_p, zoom = 0.5, order=1)

data = [cropped_vox_large, data_large_p, cropped_vox_small, data_small_p]

