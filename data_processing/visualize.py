from vtkplotter import *
import numpy as np
import SimpleITK as sitk


def read_nii(path):
    '''
    read nii.gz
    :param path: file path
    :return: 3d array
    '''
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)
    return img


def vis_nii(path):
    '''
    visualize nii.gz
    :param path: file path
    :return: None
    '''
    volume = load(path) #returns a vtkVolume object
    show(volume, bg='white', newPlotter=True)

