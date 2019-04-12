import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
import time
import torchvision.models as models
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from data_process.readDataToGAN import *
from Nets.GAN_net import Discriminator

# Hyper parameters
# G_OUTPUT_dim = 21     # generator output dimensions
INPUT_dim = 22
BATCH_SIZE = 64     # train batch size
EPOCH = 10        # iteration of all datasets
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001
LAMDA = 0.7         # PairwiseDistance leaning rate
BETA = 0.2
TEST_dim = 22
# GPU_NUM = 2

# store module filefolder

loss_records = []

SAVE_NET_PATH = os.path.join('./GANmodule/','{}'.format(time.strftime('%Y-%m-%d', time.localtime(time.time()))))


SAVE_NET_PATH_G = os.path.join(SAVE_NET_PATH,'G') # the path to save encoder network


SAVE_NET_PATH_D = os.path.join(SAVE_NET_PATH,'D')   # the path to save decoder network


def data_trans(Traindata, TrainLabel):     # transform the data to torch & Variable
    TraindataM = torch.from_numpy(Traindata).float()    # transform to float torchTensor
    Traindata_LabelM = torch.from_numpy(TrainLabel).float()
    # TestdataM = torch.from_numpy(Testdata).float()
    # TestLabelM = torch.from_numpy(TestLabel).float()
    # #Dataset
    TorchDataset = Data.TensorDataset(TraindataM, Traindata_LabelM)
    print("_______________",TraindataM.shape,TrainLabel.shape)
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=TorchDataset, batch_size=BATCH_SIZE, shuffle=True)

    return Variable(TraindataM),Variable(Traindata_LabelM),train_loader#, Variable(TestdataM),Variable(TestLabelM)

def random_generator():
    """
    func:random generator (4,1),ouput torch tensor 1,1,4,1
    :return:
    """
    #################### generator input ###################
    scaler = MinMaxScaler(feature_range=(0, 1))

    la = scaler.fit_transform(np.random.randn(4, 1))

    la = torch.from_numpy(la).float()
    la = la.expand((1, 1, 4, 1))
    return Variable(la)
    #################### generator input ###################


"""
   convtranspose2d 
    #   1.  s =1
    #     o' = i' - 2p + k - 1 
    #   2.  s >1
    # o = (i-1)*s +k-2p+n
    # n =  output_padding,p=padding,i=input dims,s=stride,k=kernel
"""

G = nn.Sequential(                      # Generator,input 1,1,4,1,output 1,64,64,21

    nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=(4, 5), stride=2, padding=1, output_padding=0,
                       bias=False),
    nn.ReLU(),

    nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(4, 5), stride=2, padding=1, output_padding=0,
                       bias=False),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
    nn.ReLU(),
)#.cuda(GPU_NUM)    #=======!!!! change for  GPU speed!!! =======#

# D = nn.Sequential(                      # Discriminator
#
#     nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),  # keep dims conv,shape: 64*22*1 to 64*22*64
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2),
#     nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),  # scale down dims conv,shape: 32*11*64 to 32*11*128
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2),
#     nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),  # scale down dims conv,shape: 16*6*128 to 16*6*256
#     nn.MaxPool2d(kernel_size=2),  # cale down dims conv,shape: 16*6*256 to 8*3*256
#     nn.ReLU(),
#     nn.Linear(1536, 300),
#     nn.Tanh(),
#     nn.Linear(300, 1),# tell the probability that the art work is made by real or fake
# )#.cuda(GPU_NUM)    #=======!!!! change for  GPU speed!!! =======#


if __name__ == '__main__':
    """
    #T or R, T represents injected message while R represents normal message
    # normal----> 'R':1 ; anomaly---> 'T':0
    
     x.permute(2,0,1)
    """
    # print(SAVE_NET_PATH)
    # print(SAVE_NET_PATH_D)
    # print(SAVE_NET_PATH_G)
    if not os.path.exists(SAVE_NET_PATH):
        os.makedirs(SAVE_NET_PATH)
    if not os.path.exists(SAVE_NET_PATH_G):
        os.makedirs(SAVE_NET_PATH_G)
    if not os.path.exists(SAVE_NET_PATH_D):
        os.makedirs(SAVE_NET_PATH_D)
    # exit()

    print('program  start at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))
    num_nor, train_all = get_data()  # num=64*10000 train_normal,train_anormal
    # num_nor, train_all = minbatch_test()  # num=64*10000 train_normal,train_anormal

    # zeros_label = np.zeros(num_anor)  # Label 0 means normal,size 1*BATCH
    # zeros = zeros_label.T
    ones_label = np.ones(num_nor)  # Label 1 means anormal,size 1*BATCH

    # train_nor_data = np.array(train_nor_data)
    # train_abnor_data = np.array(train_abnor_data)

    # print(type(train_nor_data))

    # train_all = np.vstack((train_nor_data,train_abnor_data))
    # train_all = np.random.rand(20,1,64,22)
    # exit()
    # train_all = train_all.reshape(20,64,22)

    train_all = np.expand_dims(train_all, axis=1)
    print('train data :ndim:{} dtype:{} shape:{}'.format(train_all.ndim, train_all.dtype, train_all.shape))
    # print('shape train_all:',train_all.shape, type(train_all))
    # exit()

    # tarin_all_label = np.vstack((zeros,ones))
    # train_all_label = np.append(zeros_label, ones_label, axis=0)
    train_label = np.reshape(ones_label, (-1, 1))
    print('shape:', train_label.shape, type(train_label), '\t', train_label.ndim)
    # print('shape a:', train_all_label.shape, type(train_all_label), '\t', train_all_label.ndim)
    # print(np.shape(train_all), np.shape(train_all_label))
    D = Discriminator()
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)   #D optimizer
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)   #G optimizer

    loss_func = nn.MSELoss()    # loss function
    loss_funcE = nn.CrossEntropyLoss()
    loss_func2 = nn.L1Loss()    # L1 loss 测量输入 "x" 和目标 "y" 中每个元素之间的平均绝对误差 (MAE)
    loss_P = nn.PairwiseDistance()# the p-norm

    Train , TrainL, train_loader  = data_trans(train_all, train_label)     # get ready to train datasets  #Test , TestL ,

    writer = SummaryWriter()
    # logger = Logger('./logger_logs') # set the logger
    # testLoss = np.random.randn(int(len(Test)/TEST_dim)) #save loss

    #=======!!!! change for  GPU speed!!! =======#
    ones_label = Variable(torch.ones(BATCH_SIZE))  #Label 1 means real,size 1*BATCH
    zeros_label = Variable(torch.zeros(BATCH_SIZE)) #Label 0 means fake,size 1*BATCH
    # D_label = torch.cat((zeros_label, ones_label), ).type(torch.LongTensor)

    #!!!!!!!! plot data block!!!!!!!!!#
    # plt.figure()    # plot data wave
    # plt.ion()       # coutinously
    #
    # view_noisedata = Test.data.numpy()[:30].reshape((1,3000))     # prepare data to plot
    # view_Labeldata = TestL.data.numpy()[:30].reshape((1,3000))
    #
    # ax1 = plt.subplot2grid((3,1),(0,0))         # plot noise data
    # ax3 = plt.subplot2grid((3,1),(2,0))         # plot original data
    # ax1.plot(view_noisedata[0])
    # ax3.plot(view_Labeldata[0])

    G_lossarray = [] # save loss
    # G_lossreal_array = []
    D_lossarray= []
    X = 0
    for epoch in range(EPOCH):                    # start training
        if epoch == 70:
            opt_G.param_groups[0]['lr'] = 0.00002
        for step, (T,L) in enumerate(train_loader):
            #=======!!!! change for  GPU speed!!! =======#
            T = Variable(T)
            L = Variable(L)
            # print('T.shape:',T.shape)#T.shape: torch.Size([64, 1, 64, 22])
            # print('L.shape:',L.shape)#L.shape: torch.Size([64, 1])

            G_generate = G(random_generator()) # fake painting from G (random ideas)
            # G generate tensor shape of 1,64,64,22,reshape as 64,1,64,22

            # G_generate_permute = G_generate.permute(1,0,2,3)
            # print('G_generate_permute before:',G_generate_permute.shape)#G_generate_permute before: torch.Size([64, 1, 64, 21])

            # print('G_generate shape:',G_generate.shape)#G_generate_permute before: torch.Size([64, 1, 64, 21])
            dims = L.shape[0]
            G_lables = torch.zeros((dims,1))

            # G_generate_permute = torch.cat((G_generate_permute, G_lables), 3)
            # print('G_generate_permute after:',G_generate_permute.shape)#G_generate_permute after: torch.Size([64, 1, 64, 22])

            D_real = D(T)           # D try to increase this prob
            # print('D_real.shape',D_real.shape)

            D_fake = D(G_generate)  # D try to reduce this prob
            # print('D_fake.shape',D_fake.shape)

            D_loss = -torch.mean(torch.log(D_real) + torch.log(1. - D_fake))
            D_lossarray.append(D_loss)

            # print('D_loss:',D_loss.item())

            # D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
            # G_loss = torch.mean(torch.log(1. - prob_artist1))
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # retain_variables for reusing computational graph
            opt_D.step()

            # Generator forward-loss-backward-update, update
            # for i in range(10):
            # G_generate = G(T)
            # D_fake = D(G_generate)

            # G_loss_D = nn.functional.binary_cross_entropy(D_fake, ones_label)
            G_loss_D = torch.mean(torch.log(1. - D_fake))
            # print('D_loss:',G_loss_D.item())


            # G_loss_D = -torch.mean(D_fake)
            # G_loss_D = loss_func(D_fake, ones_label)

            # G_loss_real = loss_func(G_generate_permute, L)   # add loss between generated data and Label data
            # G_loss_Pair = loss_P(G_generate_permute, L).mean()#########################################################
            # print("########!!!!!!!!",loss_P(G_generate,L))
            # if epoch > 90:
            #     G_loss = G_loss_D + G_loss_Pair
            #     # G_loss = G_loss_Pair
            # else:
            #     G_loss = G_loss_D  +  LAMDA * G_loss_Pair + BETA * (torch.max(torch.abs(G_generate_permute - L))) # +G_loss_real###############################################################
                # G_loss = G_loss_D  +  LAMDA * G_loss_Pair + BETA * ((torch.abs(G_generate.data - L.data)).max()) # +G_loss_real#
                # G_loss = LAMDA * G_loss_Pair + BETA * ((torch.abs(G_generate.data - L.data)).max()) # +G_loss_real#
            # G_loss = G_loss_real + G_loss_Pair

            # G_loss = G_loss_D + LAMDA * G_loss_Pair #+ BETA * (torch.max(G_generate - L)) #+ G_loss_Pair   18
            # G_loss = G_loss_D + LAMDA * G_loss_Pair + BETA * (torch.max(torch.abs(G_generate - L))) #+ G_loss_Pair   19
            # G_loss = G_loss_D + LAMDA * torch.sqrt(G_loss_real) + BETA * (torch.max(torch.abs(G_generate - L))) #+ G_loss_Pair   20
            # G_loss = G_loss_D + LAMDA * G_loss_Pair + BETA * (torch.max(torch.abs(G_generate_permute - L)))  # +G_loss_real###############################################################
            G_lossarray.append(G_loss_D)

            opt_G.zero_grad()
            G_loss_D.backward(retain_graph=True)
            opt_G.step()
    #==================#
            if step % 150 == 0:
                # autoencoder.eval()      # turn off dropout function ,prediction
                # print test results (test loss)
                # Gtest_data = G(Test)  #.cuda(GPU_NUM)=======!!!! change for  GPU speed!!! =======#
                # calculate the RMSE
                # Temp = (Gtest_data.data.numpy() - TestL.data.numpy()) ** 2
                # for n in range(int(len(Gtest_data)/TEST_dim)):
                #     # print('Temp shape :',np.shape(Temp))
                #     testLoss[n] = np.sqrt(np.mean(Temp[n*TEST_dim:(n+1)*TEST_dim])) # RMSE
                loss_records.append(-G_loss_D.item())
                loss_records.append(D_loss.item())

                print('Epoch: ', epoch, '|Gloss: %.4f' % (-G_loss_D.item()),'|train Dloss: %.4f'%D_loss.item())#MSE LOSS : %.6f' % G_loss_real.item(),

                # G_lossarray.append(G_loss.data[0])
                # G_lossreal_array.append(G_loss_real.data[0])
                # writer.add_scalar('G_loss', G_loss.data[0], X)
                # writer.add_scalar('D_loss', D_loss.data[0], X)
                # X = X + 1#################################################################################################################################################################
        if epoch %10== 0:
            url_G = SAVE_NET_PATH_G + '/Epoch_{}.pkl'.format(epoch)
            url_D = SAVE_NET_PATH_D + '/Epoch_{}.pkl'.format(epoch)
            torch.save(G, url_G)
            torch.save(D, url_D)

    # save Net
    Net_G_url = SAVE_NET_PATH_G +'/Net_G.pkl'
    Net_D_url = SAVE_NET_PATH_D +'/Net_D.pkl'

    torch.save(G,Net_G_url)   # save generator NET
    torch.save(D,Net_D_url)   # save generator NET
    # torch.save(D,SAVE_NET_PATH_D)
    np.savetxt(SAVE_NET_PATH +'/loss.txt',np.array(loss_records).reshape((-1,2)),fmt='%.5f',delimiter=',')
    np.savetxt(SAVE_NET_PATH_D +'/loss.txt',np.array(D_lossarray),fmt='%.5f',delimiter=',')
    np.savetxt(SAVE_NET_PATH_G +'/loss.txt',np.array(G_lossarray),fmt='%.5f',delimiter=',')

    # print(G_lossarray)
    # print(G_lossreal_array)
    print('program  end at:', time.strftime('%Y-%m-%d,%H:%M:%S', time.localtime(time.time())))