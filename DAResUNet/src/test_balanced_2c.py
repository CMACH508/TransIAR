
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nets.daresunet import DAResUNet
import torch.utils.data as Data
import pickle
from utils import cal_metrics
from torch.nn import DataParallel
import argparse

parser = argparse.ArgumentParser(description='Train neural net on .types data.')
parser.add_argument('--is_unet', type=int, default=1)
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_unet = args.is_unet
model = DAResUNet(input_channel=2)
model = DataParallel(model)
model.to(device)
if is_unet:
    model_path = '../checkpoint_2c/model_isunet1_0.8415.pth'
else:
    model_path = '../checkpoint_2c/model_isunet0_0.8415.pth'
model.load_state_dict(torch.load(model_path))
print("load done")

dataset_split = pickle.load(open('../../../dataset/dataset_cta_balanced_test.pkl','rb'))
y_test = torch.tensor(dataset_split['y_test'], dtype=torch.long)

X_test_96 = torch.tensor(dataset_split['vox_test'])[:,0:1,:,:,:]
X_test_48 = torch.tensor(dataset_split['vox_test'])[:,2:3,:,:,:]
X_test = torch.cat((X_test_96, X_test_48), dim=1)
print(X_test.shape)

test_dataset = Data.TensorDataset(X_test, y_test)
test_total = y_test.shape[0]/32
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    pred = model(voxel, is_unet=is_unet)['mid']
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

