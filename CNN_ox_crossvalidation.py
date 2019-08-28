# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from numpy import zeros, newaxis
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

print("0, 1, 2")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# Training settings
batch_size = 64
epochs = 10
learning_rate = 0.0001
nb_classes = 2
lamda = 1
sparse_threshold = 0.0005
n_splits = 10
classes = ["not superconductor", "superconductor"]
kfold = KFold(n_splits = n_splits, random_state = 42, shuffle = True)


all_fomula = np.load('all_fomula.npy')
all_ox = np.load('all_ox.npy').astype('int32')
all_idx = np.load('all_idx.npy').astype('int32')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, padding=3).cuda(device)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(device)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=1, stride=2).cuda(device)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=1).cuda(device)
        self.downsample64 = nn.Conv2d(64, 128, kernel_size=1, stride=2).cuda(device)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(device)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=1, stride=2).cuda(device)
        self.conv2_3 = nn.Conv2d(128, 256, kernel_size=1).cuda(device)
        self.downsample128 = nn.Conv2d(128, 256, kernel_size=1, stride=2).cuda(device)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(device)
        self.conv3_2 = nn.Conv2d(256, 512, kernel_size=(1, 9),  padding=(0, 4)).cuda(device)
        self.conv3_3 = nn.Conv2d(512, 512, kernel_size=(9, 1), padding=(4, 0)).cuda(device)
        self.bn64 = nn.BatchNorm2d(64).cuda(device)
        self.bn128 = nn.BatchNorm2d(128).cuda(device)
        self.bn256 = nn.BatchNorm2d(256).cuda(device)
        self.bn512 = nn.BatchNorm2d(512).cuda(device)
        self.mp0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).cuda(device)
        self.mp = nn.MaxPool2d(2).cuda(device)
        self.relu = nn.ReLU().cuda(device)
        self.fc = nn.Linear(8192, 2).cuda(device)

    def forward(self, x):
        x.cuda(device)
        x = self.mp0(self.relu(self.bn64(self.conv0(x))))
        shortcut = x

        x = self.bn64(self.conv1_1(self.relu(self.bn64(self.conv1_1(x)))))
        x = self.relu(x + shortcut)
        shortcut = x
        x = self.bn64(self.conv1_1(self.relu(self.bn64(self.conv1_1(x)))))
        x = self.relu(x + shortcut)
        shortcut = x
        x = self.bn128(self.conv1_3(self.bn64(self.conv1_2(self.relu(self.bn64(self.conv1_1(x)))))))
        shortcut = self.bn128(self.downsample64(shortcut))
        x = self.relu(x + shortcut)
        shortcut = x

        x = self.bn128(self.conv2_1(self.relu(self.bn128(self.conv2_1(x)))))
        x = self.relu(x + shortcut)
        shortcut = x
        x = self.bn128(self.conv2_1(self.relu(self.bn128(self.conv2_1(x)))))
        x = self.relu(x + shortcut)
        shortcut = x
        x = self.bn256(self.conv2_3(self.bn128(self.conv2_2(self.relu(self.bn128(self.conv2_1(x)))))))
        shortcut = self.bn256(self.downsample128(shortcut))
        x = self.relu(x + shortcut)

        x = self.bn256(self.conv3_1(self.relu(self.bn256(self.conv3_1(x)))))
        x = self.bn512(self.conv3_3(self.relu(self.bn512(self.conv3_2(x)))))



        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return F.log_softmax(x)


model = Net()
# model_fc = model.fc
if torch.cuda.is_available():
    print("cuda is available...")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        # model_fc = nn.DataParallel(model_fc)
        # model_fc.module.fc
else:
    print("cuda disabled")
model.module.fc
model.to(device)



def data_parallel(module, input, device_ids, output_device):
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# loss_function = F.nll_loss()
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda(device)
sgd_clf = SGDClassifier(max_iter = 2, random_state = 42)


def train(epoch, train_):
    model.train()
    l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for batch_idx, (data, target, idx) in enumerate(train_):
        data, target = Variable(data).float(), Variable(target)
        data = data.float().cuda(device)
        target = target.long().cuda(device)
        optimizer.zero_grad()
        output = model(data).float()
        loss = loss_function(output, target)
        # loss = F.nll_loss(output, target).cuda(device)
        # l1_regularization = torch.norm(model.fc.weight, 1).long()
        # for param in model.parameters():
        #     l1_regularization += lamda * torch.norm(param, 1).long()
        # l2_regularization += lamda * torch.norm(param, 2).long()
        loss += lamda * l1_regularization + lamda * l2_regularization
        loss.backward()
        optimizer.step()
        y_pred = torch.argmax(output, dim=1)
        if batch_idx % 10 == 0:
            train_accuracy = np.where(target.cpu() == y_pred.cpu())[0].shape[0] / target.cpu().numpy().shape[0]
            # informations
            l1_norm = 0
            l2_norm = 0
            # l1_norm = torch.norm(model.fc.weight, p=1)
            # l2_norm = torch.norm(model.fc.weight, p=2)
            # parameters = model.parameters()
            # t = abs(parameters).max() * sparse_threshold
            # nz = np.where(abs(parameters) < t)[0].shape[0]
            for y, z in zip(target.view(-1), y_pred.view(-1)):
                confusion_matrix[y.long(), z.long()] += 1
            print('epoch = {}\tLoss: {:.6f}\ttraining accuracy = {:.3}\tl1={}\tl2={}\tnz={}'.format(epoch, loss, train_accuracy, l1_norm, l2_norm, l2_norm))
            print('accuracy : {}\tprecision : {}\trecall : {}\tf1 : {}\n'.format(metrics.accuracy_score(target.cpu(), y_pred.cpu()), metrics.precision_score(target.cpu(), y_pred.cpu()), metrics.recall_score(target.cpu(), y_pred.cpu()), metrics.f1_score(target.cpu(), y_pred.cpu())))
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_.dataset), 100. * batch_idx / len(train_), loss))
    print(confusion_matrix)


def test(test_):
    model.eval()
    test_loss = 0
    correct = 0
    l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    global loss_list
    global accuracy_list
    global confusion
    for data, target, idx in test_:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        data = data.float().cuda(device)
        target = target.long().cuda(device)
        output = model(data)
        test_loss += loss_function(output, target)
        # for param in model.parameters():
        #     l1_regularization += lamda * torch.norm(param, 1).long()
        # l2_regularization += lamda * torch.norm(param, 2).long()
        test_loss += l1_regularization + l2_regularization
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for y, z in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[y.long(), z.long()] += 1
        # sum up batch loss
        print(idx)
    test_loss /= len(test_.dataset)
    # informations
    l1_norm = 0
    l2_norm = 0
    # l1_norm = torch.norm(model.fc.weight, p=1)
    # l2_norm = torch.norm(model.fc.weight, p=2)
    # parameters = model.parameters()
    # t = abs(parameters).max() * sparse_threshold
    # nz = np.where(abs(parameters) < t)[0].shape[0]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_.dataset),
        100. * correct / len(test_.dataset)))
    test_loss = test_loss.cpu().detach().numpy()
    loss_list = np.append(loss_list, test_loss)
    accuracy_list = np.append(accuracy_list, 100. * correct / len(test_.dataset))
    if not confusion.any():
        confusion = confusion_matrix.numpy()
    else:
        confusion = np.append(confusion, confusion_matrix.numpy(), axis = 0)
    print(confusion_matrix)
    print(confusion)

    # print(confusion)


idx = 0
acc_list = np.zeros((n_splits, epochs))
lss_list = np.zeros((n_splits, epochs))
confusion = np.zeros(0)
for train_index, test_index in kfold.split(all_fomula, all_ox):
    clone_clf = clone(sgd_clf)
    fomula_train_folds = all_fomula[train_index]
    ox_train_folds = all_ox[train_index]
    idx_train_folds = all_idx[train_index]
    fomula_test_fold = all_fomula[test_index]
    ox_test_fold = all_ox[test_index]
    idx_test_fold = all_idx[test_index]
    print(train_index)
    print(test_index)

    class MyDataset1(Dataset):
        def __init__(self):
            self.len = len(fomula_train_folds)
            self.x_data = torch.from_numpy(fomula_train_folds)
            self.y_data = torch.from_numpy(ox_train_folds)
            self.z_idx = torch.from_numpy(idx_train_folds)

        def __getitem__(self, index):
            img = self.x_data[index]
            label = self.y_data[index]
            idx = self.z_idx[index]
            return(img, label, idx)

        def __len__(self):
            count = self.len
            return count
    dataset_train = MyDataset1()
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    class MyDataset2(Dataset):
        def __init__(self):

            self.len = len(fomula_test_fold)
            self.x_data = torch.from_numpy(fomula_test_fold)
            self.y_data = torch.from_numpy(ox_test_fold)
            self.z_idx = torch.from_numpy(idx_test_fold)

        def __getitem__(self, index):
            img = self.x_data[index]
            label = self.y_data[index]
            idx = self.z_idx[index]
            return(img, label, idx)

        def __len__(self):
            count = self.len
            return count
    dataset_test = MyDataset2()
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    loss_list = np.zeros(0)
    accuracy_list = np.zeros(0)
    for epoch in range(epochs):
        train(epoch, train_loader)
        test(test_loader)
    for i in range(epochs):
        lss_list[idx][i] = loss_list[i]
        acc_list[idx][i] = accuracy_list[i]
    idx += 1



for i in range(n_splits):
    for j in range(epochs):
        print('{}번쨰 fold {}번째 epoch의 loss: {:.5f}, accuracy: {:.3f}%'.format(i + 1, j + 1, lss_list[i][j], acc_list[i][j]))
        print('confusion matrix : \n{}\n{}'.format(confusion[2*i*epochs + j], confusion[2*i*epochs + j + 1]))
