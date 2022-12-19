import os.path
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as Data
import pickle
from os.path import join
import os
import sys
import datetime
import numpy as np
import argparse
from nets.daresunet import DAResUNet
from torch.nn import DataParallel

parser = argparse.ArgumentParser(description='Train neural net on .types data.')
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--is_unet', type=int, default=1)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-f', '--factor', type=float, default=1)
args = parser.parse_args()
print(args)

is_unet = args.is_unet
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def train(n_epoch=200, BS=128, LR=1e-4):
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0")

    input_shape = (48, 48, 48)

    dataset_train = pickle.load(open('../../dataset/dataset_cta_balanced_train.pkl', 'rb'))
    dataset_test = pickle.load(open('../../dataset/dataset_cta_balanced_test.pkl', 'rb'))

    X_train_96 = torch.tensor(dataset_train['vox_train'])[:,0:1,:,:,:]
    X_test_96 = torch.tensor(dataset_test['vox_test'])[:,0:1,:,:,:]
    X_train_48 = torch.tensor(dataset_train['vox_train'])[:,2:3,:,:,:]
    X_test_48 = torch.tensor(dataset_test['vox_test'])[:,2:3,:,:,:]
    X_train = torch.cat((X_train_96, X_train_48), dim=1)
    X_test = torch.cat((X_test_96, X_test_48), dim=1)
    y_train = torch.tensor(dataset_train['y_train'], dtype=torch.long)
    y_test = torch.tensor(dataset_test['y_test'], dtype=torch.long)
    print(X_train.shape)
    print(X_test.shape)

    train_dataset = Data.TensorDataset(X_train, y_train)
    test_dataset = Data.TensorDataset(X_test, y_test)

    train_total = y_train.shape[0]
    test_total = y_test.shape[0] / 32

    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # encoder = Net(n_classes=2, input_shape=input_shape)
    # print(model)
    model = DAResUNet(input_channel=2)

    train_acc_list = []
    test_acc_list = []
    train_acc_each_list = []
    test_acc_each_list = []

    optimizer = optim.Adam(model.parameters(), lr=LR)
    model = DataParallel(model)
    model.to(device)

    best_acc = 0
    reconstruct_loss = 0
    for epoch in range(n_epoch):
        print()
        print("{} start epoch: {} lr: {}".format(get_time(), epoch, optimizer.param_groups[0]['lr']))
        train_correct = 0
        train_correct_each = [0 for c in range(2)]
        train_total_each = [0 for c in range(2)]

        for i, (voxel, cls_idx) in enumerate(train_dataloader, 0):
            voxel, cls_idx = voxel.to(device), cls_idx.to(device)
            voxel = voxel.float()

            optimizer.zero_grad()
            model = model.train()
            pred = model(voxel, is_unet=is_unet)

            cls_loss = F.cross_entropy(pred['mid'], cls_idx)
            loss = cls_loss
            if is_unet:
                # print('y:{} input:{}'.format(pred['y'].shape, voxel.shape))
                reconstruct_loss = F.mse_loss(pred['y'], voxel) * args.factor
                loss = cls_loss + reconstruct_loss
                reconstruct_loss = reconstruct_loss.item()

            loss.backward()
            optimizer.step()

            pred_choice = pred['mid'].max(1)[1]
            correct = pred_choice.eq(cls_idx).cpu().sum()
            train_correct += correct

            for number in range(2):
                train_correct_each[number] += torch.sum(pred_choice[cls_idx == number] == number).item()
                train_total_each[number] += cls_idx[cls_idx == number].shape[0]

        train_acc = train_correct / float(train_total)
        train_acc_list.append(train_acc)

        train_acc_each = [train_correct_each[c] / train_total_each[c] for c in range(2)]
        train_acc_each_list.append(train_acc_each)

        print("cls_loss:{:.3f}, recon_loss:{:.3f}, all_loss:{:.3f}, train acc:{:.4f}, [0 acc:{:.4f}, 1 acc:{:.4f}]".format(
            cls_loss.item(), reconstruct_loss, loss.item(), train_acc, train_acc_each[0], train_acc_each[1]))

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
            pred_choice = pred.max(0)[1]

            correct_t = 0
            if pred_choice == cls_idx[0]:
                correct_t = 1
            test_correct += correct_t

            test_correct_each[cls_idx[0]] += correct_t
            test_total_each[cls_idx[0]] += 1

        test_acc = test_correct / float(test_total)
        test_acc_list.append(test_acc)
        test_acc_each = [test_correct_each[c] / test_total_each[c] for c in range(2)]
        test_acc_each_list.append(test_acc_each)

        print("test acc:{:.3f}, [0 acc:{:.4f}, 1 acc:{:.4f}]".format(test_acc, test_acc_each[0], test_acc_each[1]))

        if test_acc > best_acc:
            print('Saving.. test_acc={:.4f}'.format(test_acc))
            torch.save(model.state_dict(), '../checkpoint/model_{}.pth'.format(test_acc))
            best_acc = test_acc


if __name__ == '__main__':
    sys.path.append(os.path.abspath('./'))
    train(n_epoch=50, BS=args.batch_size, LR=1e-4)


