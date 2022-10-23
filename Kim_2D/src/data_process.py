import numpy as np
import cv2


def proj(image3d, axis=0, direction=0):
    '''
    project 3D image to 2D
    :param image3d: 3D numpy array (x, y, z)
    :param axis: 0 for x, 1 for y, 2 for z
    :param direction: 0 for 0->n, 1 for n->0
    :return: 2D numpy array
    '''

    shape3d = image3d.shape
    axes = list(range(3))
    axes.pop(axis)
    shape2d = (shape3d[axes[0]], shape3d[axes[1]])
    ret = np.zeros(shape2d)
    num_slices = shape3d[axis]
    for i in range(num_slices):
        cur = None
        index = i if direction == 0 else num_slices - 1 - i
        if axis == 0:
            cur = image3d[index, :, :]
        elif axis == 1:
            cur = image3d[:, index, :]
        elif axis == 2:
            cur = image3d[:, :, index]
        ret = np.where(cur > 0, cur, ret)
    return ret


def proj_six(image3d):
    ret = []
    for i in range(3):
        for j in range(2):
            proj_x = proj(image3d, i, j)
            proj_x = cv2.resize(proj_x, (224, 224), cv2.INTER_CUBIC)
            proj_x = cv2.cvtColor(proj_x.astype(np.float32), cv2.COLOR_GRAY2RGB)
            ret.append(proj_x.transpose(2, 0, 1))
    return ret


def data_augmentation(image2d):
    ret = [image2d]
    image2d_ = image2d.transpose(1, 2, 0)
    # Flipped Horizontally
    h_flip = cv2.flip(image2d_, 1)
    ret.append(h_flip.transpose(2, 0, 1))

    # Flipped Vertically
    v_flip = cv2.flip(image2d_, 0)
    ret.append(v_flip.transpose(2, 0, 1))

    # Flipped Horizontally & Vertically
    hv_flip = cv2.flip(image2d_, -1)
    ret.append(hv_flip.transpose(2, 0, 1))

    return ret


if __name__ == '__main__':
    cube = np.random.rand(48,48,48)
    # project cube from six directions
    proj_list = proj_six(cube)
    # 2D image augmentation
    proj_aug = data_augmentation(proj_list[0])