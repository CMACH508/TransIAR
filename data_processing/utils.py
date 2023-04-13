import numpy as np
import SimpleITK as sitk
import copy


def read_nii(file_path):
    '''
    :param file: file path
    :return: 3d array
    '''
    itk_img = sitk.ReadImage(file_path)
    img = sitk.GetArrayFromImage(itk_img)
    return img


def save_nii(img, path):
    '''
    :param img: 3d array
    :param path: file path
    :return: None
    '''
    out = sitk.GetImageFromArray(img)
    sitk.WriteImage(out, path)


def crop(id, size = 32):
    '''
    :param id: file name
    :param size: cube size
    :return: array:(32,32,32)
    '''
    img = read_nii('./label/{}.nii.gz'.format(id))
    data = np.argwhere(img==1)
    boundary = []
    for i in range(3):
        axis = data[:,i]
        middle = int((max(axis) + min(axis)) / 2)
        left = int(middle - size/2 + 1)
        right = int(middle + size/2) + 1
        boundary.append(left)
        boundary.append(right)
    path = './CTA/{}.nii.gz'.format(id)
    img_with_v = read_nii(path)
    return img_with_v[boundary[0]:boundary[1],boundary[2]:boundary[3],boundary[4]:boundary[5]]


def bfs(cube, thresh1 = 0.6):
    '''
    :param cube: 3d array
    :return: 3d array
    '''
    thresh2 = 0.1
    cube_tmp = copy.deepcopy(cube)
    tmp = cube.reshape(-1)
    maxn = max(tmp)
    cube = cube / maxn
    size = cube.shape[0]
    visited = np.zeros((size,size,size),dtype=int)
    queue = []
    step = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    curx = int(size / 2)
    cury = int(size / 2)
    curz = int(size / 2)
    queue.append([curx, cury, curz])
    visited[curx][cury][curz] = 1
    while len(queue) != 0:
        curpos = queue.pop(0)
        for d in range(6):
            i = curpos[0]+step[d][0]
            j = curpos[1]+step[d][1]
            k = curpos[2]+step[d][2]
            if i<0 or i>=size or j<0 or j>=size or k<0 or k>=size or visited[i][j][k]==1 or cube[i][j][k]<thresh1:
                continue
            queue.append([i, j, k])
            visited[i][j][k] = 1

    for i in range(size):
        for j in range(size):
            for k in range(size):
                if visited[i][j][k] == 1:
                    queue.append([i,j,k])
    iter = 2
    for i in range(iter):
        queue_tmp = []
        for p in queue:
            for d in range(6):
                i = p[0]+step[d][0]
                j = p[1]+step[d][1]
                k = p[2]+step[d][2]
                if i<0 or i>=size or j<0 or j>=size or k<0 or k>=size or visited[i][j][k]==1 or cube[i][j][k]<thresh2:
                    continue
                queue_tmp.append([i, j, k])
                visited[i][j][k] = 1
        queue = queue_tmp

    num = sum(visited.reshape(-1))
    # print(num)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if visited[i][j][k] == 0:
                    cube_tmp[i][j][k] = 0

    return cube_tmp, visited, num


def process(cube, thresh = 0.6):
    '''
    :param cube: 3d array
    :param thresh: float
    :return: 3d array
    '''
    cube_p, visited, num = bfs(cube, thresh)
    size = cube.shape[0]
    flag = sum(visited[0,:,:].reshape(-1)) + sum(visited[size-1,:,:].reshape(-1)) +\
           sum(visited[:,0,:].reshape(-1)) + sum(visited[:,size-1,:].reshape(-1)) +\
           sum(visited[:,:,0].reshape(-1)) + sum(visited[:,:,size-1].reshape(-1))
    while num < 125 or flag < 1:
        thresh -= 0.05
        cube_p, visited, num = bfs(cube, thresh)
        flag = sum(visited[0,:,:].reshape(-1)) + sum(visited[size-1,:,:].reshape(-1)) +\
           sum(visited[:,0,:].reshape(-1)) + sum(visited[:,size-1,:].reshape(-1)) +\
           sum(visited[:,:,0].reshape(-1)) + sum(visited[:,:,size-1].reshape(-1))
    return cube_p


