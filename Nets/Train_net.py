import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from data_process.readDataToNets import get_data
from Nets.GAN_net import Discriminator
# from ECG_exp.logger import Logger
# from ECG_exp02.data_process import read_data
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# from graphviz import Digraph

# Hyper parameters
train_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/merge_data/train/"
test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/merge_data/test/"

train_normal = train_addr + 'copy_normal.txt'
train_anormal = train_addr + 'copy_anormal.txt'
BATCH_SIZE = 64     # train batch size
EPOCH = 5       # iteration of all datasets
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001
# GPU_NUM = 2

# train_nor_data, train_abnor_data = get_data()
num_nor, num_anor, train_all = get_data(train_normal,train_anormal,num=64*10000)
zeros_label = np.zeros(num_nor) #Label 0 means normal,size 1*BATCH
# zeros = zeros_label.T
ones_label = np.ones(num_anor) #Label 1 means anormal,size 1*BATCH

# train_nor_data = np.array(train_nor_data)
# train_abnor_data = np.array(train_abnor_data)

# print(type(train_nor_data))

# train_all = np.vstack((train_nor_data,train_abnor_data))
# train_all = np.random.rand(20,1,64,22)
# exit()
# train_all = train_all.reshape(20,64,22)

train_all = np.expand_dims(train_all, axis=1)
print('data :ndim:{} dtype:{} shape:{}'.format(train_all.ndim, train_all.dtype, train_all.shape))
# print('shape train_all:',train_all.shape, type(train_all))
# exit()

# tarin_all_label = np.vstack((zeros,ones))
train_all_label = np.append(zeros_label,ones_label,axis=0)

print('shape:',train_all_label.shape, type(train_all_label),'\t',train_all_label.ndim)

train_all_label = np.reshape(train_all_label,(-1,1))

print('shape a:',train_all_label.shape,type(train_all_label),'\t',train_all_label.ndim)
# exit()
print(np.shape(train_all), np.shape(train_all_label))

SAVE_NET_PATH = './GAN_G_20190327.pkl'   # the path to save encoder network
# SAVE_NET_PATH_G = '/root/NN_saved/GAN_G_20190327.pkl'   # the path to save encoder network
# SAVE_NET_PATH_D = '/root/NN_saved/GAN_D_20190327.pkl'   # the path to save encoder network

def data_trans():     # transform the data to torch & Variable
    Traindata_LabelM = torch.from_numpy(train_all_label).float()
    TraindataM = torch.from_numpy(train_all).float()  # transform to float torchTensor
    # TorchDataset = Data.TensorDataset(data_tensor=TraindataM, target_tensor=Traindata_LabelM)
    TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    return Variable(TraindataM),Variable(Traindata_LabelM),train_loader

D_net = Discriminator()
opt_D = torch.optim.Adam(D_net.parameters(), lr=LR_D)   #D optimizer

loss_func = nn.MSELoss()    # loss function

Train , TrainL , train_loader = data_trans()     # get ready to train datasets
writer = SummaryWriter()

#=======!!!! change for  GPU speed!!! =======#


G_lossarray = [] # save loss
G_lossreal_array = []
X = 0
for epoch in range(EPOCH):                    # start training
    for step, (T,L) in enumerate(train_loader):
        #=======!!!! change for  GPU speed!!! =======#
        T = Variable(T)#.cuda(GPU_NUM)
        L = Variable(L)#.cuda(GPU_NUM)

        D_generate = D_net(T)       # fake painting from G (random ideas)

        D_loss = loss_func(D_generate, L)

        opt_D.zero_grad()
        D_loss.backward()  # retain_variables for reusing computational graph
        opt_D.step()

        if step % 10 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % D_loss.item())

    if (epoch+1)%5 == 0:
        torch.save(D_net, SAVE_NET_PATH)

# save Net
torch.save(D_net, SAVE_NET_PATH)  # save generator NET

# print(G_lossarray)
# print(G_lossreal_array)