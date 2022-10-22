
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import TransIAR_AF
import torch.utils.data as Data
import pickle


def train(n_epoch = 200, BS = 128, LR=1e-4, step_size=200, dim = 256, depth = 4, heads = 4, dim_head = 256, mlp_dim = 1024):
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape = (48,48,48)
    print("Loading data...")
    dataset_train = pickle.load(open('../dataset/dataset_cta_balanced_train.pkl','rb'))
    dataset_test = pickle.load(open('../dataset/dataset_cta_balanced_test.pkl', 'rb'))

    X_train = torch.tensor(dataset_train['vox_train'])
    X_test = torch.tensor(dataset_test['vox_test'])
    y_train = torch.tensor(dataset_train['y_train'], dtype=torch.long)
    y_test = torch.tensor(dataset_test['y_test'], dtype=torch.long)
    print(X_train.shape)
    print(X_test.shape)

    idx_af_train = pickle.load(open('../dataset/dataset_af_balanced_train.pkl','rb'))
    idx_af_test = pickle.load(open('../dataset/dataset_af_balanced_test.pkl', 'rb'))
    af_train = idx_af_train['chara_train']
    af_test = idx_af_test['chara_test']
    af_train = torch.tensor(af_train, dtype=torch.float)
    af_test = torch.tensor(af_test, dtype=torch.float)
    print(af_train.shape)
    print(af_test.shape)

    train_dataset = Data.TensorDataset(X_train, af_train, y_train)
    test_dataset = Data.TensorDataset(X_test, af_test, y_test)
    train_total = y_train.shape[0]
    test_total = y_test.shape[0]/32
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TransIAR_AF(n_classes=2, input_shape = input_shape, dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim)
    print(model)

    train_acc_list = []
    test_acc_list = []
    train_acc_each_list = []
    test_acc_each_list = []

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    model.to(device)

    best_acc = 0

    for epoch in range(n_epoch):
        print("epoch: {} lr: {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_correct = 0
        train_correct_each = [0 for c in range(2)]
        train_total_each = [0 for c in range(2)]

        for i, (voxel, af, cls_idx) in enumerate(train_dataloader, 0):
            voxel, af, cls_idx = voxel.to(device), af.to(device), cls_idx.to(device)
            voxel = voxel.float()

            optimizer.zero_grad()
            model = model.train()
            pred = model(voxel, af)
            loss = F.cross_entropy(pred, cls_idx)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            train_correct += correct

            for number in range(2):
                train_correct_each[number] += torch.sum(pred_choice[cls_idx == number] == number).item()
                train_total_each[number] += cls_idx[cls_idx == number].shape[0]

        train_acc = train_correct / float(train_total)
        train_acc_list.append(train_acc)

        train_acc_each = [train_correct_each[c]/train_total_each[c] for c in range(2)]
        train_acc_each_list.append(train_acc_each)

        print("train accuracy: {}".format(train_acc))
        print("0 accuracy: {}, 1 accuracy: {}".format(train_acc_each[0], train_acc_each[1]))

        scheduler.step()

        test_correct = 0
        test_correct_each = [0 for c in range(2)]
        test_total_each = [0 for c in range(2)]

        for i, (voxel, af, cls_idx) in enumerate(test_dataloader, 0):
            voxel, af, cls_idx = voxel.to(device), af.to(device), cls_idx.to(device)
            voxel = voxel.float()
            model = model.eval()
            pred = model(voxel, af)
            pred = F.softmax(pred, dim=1)
            pred = pred.sum(dim=0)
            pred_choice = pred.data.max(0)[1]

            correct_t = 0
            if pred_choice == cls_idx[0]:
                correct_t = 1
            test_correct += correct_t

            test_correct_each[cls_idx[0]] += correct_t
            test_total_each[cls_idx[0]] += 1

        test_acc = test_correct / float(test_total)
        test_acc_list.append(test_acc)
        test_acc_each = [test_correct_each[c]/test_total_each[c] for c in range(2)]
        test_acc_each_list.append(test_acc_each)

        print("test accuracy: {}".format(test_acc))
        print("0 accuracy: {}, 1 accuracy: {}".format(test_acc_each[0], test_acc_each[1]))

        if test_acc > best_acc:
            print('Saving..')
            # torch.save(model.state_dict(), '../checkpoint/model_{}.pth'.format(test_acc))
            best_acc = test_acc

        print("best test accuracy: {}".format(best_acc))


if __name__ == "__main__":
    n_epoch = 100
    BS = 128
    LR=1e-4
    step_size=100
    dim = 256
    depth = 4
    heads = 4
    dim_head = 256
    mlp_dim = 1024
    train(n_epoch, BS, LR, step_size, dim, depth, heads, dim_head, mlp_dim)
