# train the GAN model
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from net_model.generator import Generator
from net_model.discriminator import Discriminator
# from eval import eval_net

# hyper parameters
data_path = 'dataset/QT_data/QT_add_klg_train400_5channel/'
dir_checkpoint = 'dataset/net_checkpoint_QT/add_klg_5channel_noMax_GAN_MSE_deeper_lrG001/'
BATCH_SIZE = 64
EPOCHS = 3000
LR_G = 0.001
LR_D = 0.001
GPU_NUM = 2
LAMDA = 0.5
BETA = 0.3
EVL = True


def pre_data(path):
    train_data = scio.loadmat(path + 'add_klg_train_sig1_02.mat')
    train_data = train_data['train']
    train_label = scio.loadmat(path + 'QT_add_klg_train_label_sig1.mat')
    train_label = train_label['label']
    train_label = np.expand_dims(train_label, axis=1)
    # test_data = scio.loadmat(path + 'new_test.mat')
    # test_data = test_data['test']
    # test_label = scio.loadmat(path + 'new_test_label.mat')
    # test_label = test_label['test_label']
    # val_data = scio.loadmat(path + 'new_val.mat')
    # val_data = val_data['val']
    # val_label = scio.loadmat(path + 'new_val_label.mat')
    # val_label = val_label['val_label']
    # print(np.shape(val_data))

    Traindata = torch.from_numpy(train_data).float()    # transform to float torchTensor
    Traindata_Label = torch.from_numpy(train_label).float()

    # Testdata = torch.from_numpy(test_data).float()
    # TestLabel = torch.from_numpy(test_label).float()
    #
    # Valdata = torch.from_numpy(val_data).float()
    # ValLabel = torch.from_numpy(val_label).float()

    #Dataset
    TorchDataset = Data.TensorDataset(data_tensor=Traindata, target_tensor=Traindata_Label)
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    return Variable(Traindata),Variable(Traindata_Label), train_loader

Train , Train_L , train_loader = pre_data(data_path)

writer = SummaryWriter()

GNet = Generator().cuda(GPU_NUM)
DNet = Discriminator().cuda(GPU_NUM)

optimizer_G = torch.optim.Adam(GNet.parameters(), lr= LR_G)
optimizer_D = torch.optim.Adam(DNet.parameters(), lr= LR_D)

loss_G = nn.MSELoss()
loss_G_pair = nn.PairwiseDistance()
loss_D = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    gloss = 0   # observe loss
    dloss = 0

    # evaluation
    # if EVL:
    #     evl_value = eval_net(GNet , Val , Val_L)
    #     print('Validation MSE: {}'.format(evl_value))
    #     writer.add_scalar('Validation MSE', evl_value, epoch)

    for step, (X,Y) in enumerate(train_loader):

        X = Variable(X).cuda(GPU_NUM)
        Y = Variable(Y).cuda(GPU_NUM)

        G_gen = GNet(X)

        D_real = DNet(Y)
        D_fake = DNet(G_gen)

        # update Discriminator

        ones_label = Variable(torch.LongTensor(np.ones(len(Y)))).cuda(GPU_NUM)  # Label 1 means real,size 1*BATCH
        zeros_label = Variable(torch.LongTensor(np.zeros(len(Y)))).cuda(GPU_NUM)  # Label 0 means fake,size 1*BATCH

        D_loss_real = loss_D(D_real,ones_label)

        # optimizer_D.zero_grad()
        # D_loss_real.backward(retain_variables=True)
        # optimizer_D.step()

        D_loss_fake = loss_D(D_fake,zeros_label)

        D_loss = D_loss_fake + D_loss_real

        optimizer_D.zero_grad()
        D_loss.backward(retain_variables=True)
        optimizer_D.step()

        dloss += D_loss.data[0]

        # XX = np.squeeze(G_gen.data.cpu().numpy(), axis=1)
        # G_gen.data = torch.from_numpy(XX).float().cuda(GPU_NUM)
        # # print('jjjjjj',G_gen.data.cpu(),np.shape(G_gen.data.cpu().numpy()))
        # YY = np.squeeze(Y.data.cpu().numpy(), axis=1)
        # Y.data = torch.from_numpy(YY).float().cuda(GPU_NUM)

        # update Generator
        G_loss = loss_G(G_gen , Y) + loss_D(D_fake, ones_label)
        # G_loss = loss_D(D_fake, ones_label) + #LAMDA * torch.mean(loss_G_pair(G_gen , Y)) #+ BETA * torch.max(torch.abs(G_gen - Y))
        gloss += G_loss.data[0]

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(epoch,'Epoch finished ! G_Loss: {:.6f} D_Loss: {:.6f}'.format(gloss / step, dloss / step))
    writer.add_scalar('G_loss', gloss / step , epoch)
    writer.add_scalar('D_loss', dloss / step , epoch)

    if (epoch + 1) % 50 == 0:
        if not os.path.exists(dir_checkpoint):
            os.makedirs(dir_checkpoint)
        torch.save(GNet.state_dict(),dir_checkpoint + 'G_length1000_{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))