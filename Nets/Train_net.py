import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from data_process.readDataToGAN import *
from Nets.GAN_net import Discriminator
# from ECG_exp.logger import Logger
# from ECG_exp02.data_process import read_data
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# from graphviz import Digraph
import os
import time

print('program run at:','program run at:',time.strftime('%Y-%m-%d,%H:%M:%S',time.localtime(time.time())))

def truncted_point(loss,count = 0):
    """#如果一个损失在15个 step 都保持在小数点后第四位数为0,训练退出"""

    if int(loss*10000) == 0:
        count += 1
    else:
        count = 0
    if count ==15:
        save_url = SAVE_NET_PATH + '{}_Epoch_{}.pkl'.format(time.strftime('%Y-%m-%d',time.localtime(time.time())),epoch)
        torch.save(D_net, save_url)

# train_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/train/"

# train_normal = train_addr + 'train_normal.txt'
# train_anormal = train_addr + 'train_anormal.txt'
BATCH_SIZE = 64     # train batch size
EPOCH = 100      # iteration of all datasets
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001


def data_trans():     # transform the data to torch & Variable
    Traindata_LabelM = torch.from_numpy(train_all_label).float()
    TraindataM = torch.from_numpy(train_all).float()  # transform to float torchTensor
    # TorchDataset = Data.TensorDataset(data_tensor=TraindataM, target_tensor=Traindata_LabelM)
    TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    return Variable(TraindataM),Variable(Traindata_LabelM),train_loader


if __name__ == '__main__':
    """get data"""
    # num_nor, num_anor, train_all = get_data(keyword='train')#num=64*10000 train_normal,train_anormal

    labels, train_all = getTrainDiscriminor(mark='train')  # added
    # zeros_label = np.zeros(num_nor) #Label 0 means normal,size 1*BATCH
    # # zeros = zeros_label.T
    # ones_label = np.ones(num_anor) #Label 1 means anormal,size 1*BATCH

    # train_nor_data = np.array(train_nor_data)
    # train_abnor_data = np.array(train_abnor_data)

    # print(type(train_nor_data))

    # train_all = np.vstack((train_nor_data,train_abnor_data))
    # train_all = np.random.rand(20,1,64,22)
    # exit()
    # train_all = train_all.reshape(20,64,22)

    train_all = np.expand_dims(train_all, axis=1)
    # print('data :ndim:{} dtype:{} shape:{}'.format(train_all.ndim, train_all.dtype, train_all.shape))
    # print('shape train_all:',train_all.shape, type(train_all))
    # exit()

    # tarin_all_label = np.vstack((zeros,ones))
    labels = np.array(labels)  # added
    # train_all_label = np.append(zeros_label,ones_label,axis=0)
    train_all_label = np.reshape(labels, (-1, 1))  # added
    print('data shape:{},labels shape'.format(train_all.shape, train_all_label.shape))  # added

    # print('train_all_label shape:',train_all_label.shape, type(train_all_label),'\t',train_all_label.ndim)

    """solve address"""
    current_dir = './Dmodule/{}'.format(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    os.chdir(current_dir)

    SAVE_NET_PATH = './module/'  # the path to save encoder network

    if not os.path.exists(SAVE_NET_PATH):
        os.makedirs(SAVE_NET_PATH)

    """load parameter"""
    D_net = Discriminator()
    opt_D = torch.optim.Adam(D_net.parameters(), lr=LR_D)   #D optimizer
    loss_func = nn.MSELoss()    # loss function
    # loss_func = nn.CrossEntropyLoss()    # loss function

    Train , TrainL , train_loader = data_trans()     # get ready to train datasets
    writer = SummaryWriter(log_dir='./logs/')

    X = 0
    # count = 0

    """train"""
    for epoch in range(EPOCH):                    # start training

        for step, (T,L) in enumerate(train_loader):
            #=======!!!! change for  GPU speed!!! =======#
            T = Variable(T)#.cuda(GPU_NUM)
            L = Variable(L)#.cuda(GPU_NUM)

            D_generate = D_net(T)       # fake painting from G (random ideas)

            D_loss = loss_func(D_generate, L)
            X += 1
            # writer.add_scalar('train_loss',D_loss, epoch)
            writer.add_scalar('train_loss',D_loss, X)
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # retain_variables for reusing computational graph
            opt_D.step()

            if step % 10 == 0:
                print('Epoch: ', epoch, '| train loss: %.6f' % D_loss.item())

        if (epoch+1)%5 == 0:
            # count += 1
            save_url = SAVE_NET_PATH + 'Epoch_{}.pkl'.format(epoch)
            torch.save(D_net, save_url)
    #save graph
    writer.add_graph(D_net,(Train,))

    # save Net
    # torch.save(D_net, SAVE_NET_PATH)  # save generator NET
    writer.close()

    # print(G_lossarray)
    # print(G_lossreal_array)