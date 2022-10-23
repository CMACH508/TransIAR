
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from AlexNet_v2 import AlexNet
import torch.utils.data as Data
import pickle
from utils import cal_metrics
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Train neural net on IA data.')
parser.add_argument('--is_bfs', type=int, default=1)
args = parser.parse_args()
BFS = args.is_bfs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNet()
model.to(device)

if BFS:
    model_path = '../checkpoint/model_bfs.pth'
    dataset_split = pickle.load(open('../../dataset_2d/dataset_2d_test_bfs.pkl','rb'))
else:
    model_path = '../checkpoint/model.pth'
    dataset_split = pickle.load(open('../../dataset_2d/dataset_2d_test.pkl','rb'))

model.load_state_dict(torch.load(model_path))
print("load done")

X_test = torch.tensor(np.array(dataset_split['X_test']))
y_test = torch.tensor(dataset_split['y_test'], dtype=torch.long)

test_dataset = Data.TensorDataset(X_test, y_test)
test_total = y_test.shape[0]/6
print(test_total)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)

y_true = []
y_pred = []
y_prob = []

test_correct = 0
test_correct_each = [0 for c in range(2)]
test_total_each = [0 for c in range(2)]

for i, (voxel, cls_idx) in enumerate(test_dataloader, 0):
    voxel, cls_idx = voxel.to(device), cls_idx.to(device)
    voxel = voxel.float()
    model = model.eval()
    pred = model(voxel)
    pred = F.softmax(pred, dim=1)
    pred = pred.sum(dim=0)
    pred_choice = pred.data.max(0)[1]
    pred_choice = pred_choice.cpu().detach().numpy()
    cls_idx = cls_idx.cpu().detach().numpy()

    prob = (pred/32).cpu().detach().numpy()
    prob = prob[1]
    y_prob.append(prob)

    y_pred.append(pred_choice)
    y_true.append(cls_idx[0])

    correct_t = 0
    if pred_choice == cls_idx[0]:
        correct_t = 1
    test_correct += correct_t

    test_correct_each[cls_idx[0]] += correct_t
    test_total_each[cls_idx[0]] += 1

test_acc = test_correct / float(test_total)
test_acc_each = [test_correct_each[c]/test_total_each[c] for c in range(2)]

print("test accuracy: {}".format(test_acc))
print("0 accuracy: {}, 1 accuracy: {}".format(test_acc_each[0], test_acc_each[1]))

cal_metrics(y_true, y_pred, y_prob)