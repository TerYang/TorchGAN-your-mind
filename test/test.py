# test the generator network
import os
import numpy as np
import torch
import time
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.autograd import Variable
# from net_model.autoencoder import AutoEncoder
from Nets.GAN_net import Discriminator
from data_process.readDataToNets import get_data
Net_PATH = '../Nets/module/2019-03-28_Epoch_29.pkl'

# test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/merge_data/test/"
test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/test/"
t1 = time.time()

test_normal = test_addr + 'test_normal.txt'
test_anormal = test_addr + 'test_anormal.txt'

# rows = 1
# PATH = 'dataset/QT_data/QT_add_klg_train400_4channel/'

# Net_PATH = 'dataset/net_checkpoint_new/length400/'
# Predictions = 'dataset/predictions_new_QT/add_klg_length400_3channel_02_AE_deeper_MSE_3000_lr001/'
# GPU_NUM = 0

# Test = scio.loadmat(PATH + 'QT_add_klg_test_sig1.mat')
# Label = scio.loadmat(PATH + 'QT_add_klg_test_label_sig1.mat')
# Train = scio.loadmat(PATH + 'new_train.mat')
# Train_label = scio.loadmat(PATH + 'new_train_label.mat')
#
# Train = Train['train']
# # Train = np.expand_dims(Train, axis=1)
# Train_label = Train_label['train_label']
# # Train_label = np.expand_dims(Train_label, axis=1)
# Train_data = Variable(torch.from_numpy(Train).float())
# Train_label = Variable(torch.from_numpy(Train_label).float())
t_nor, t_anor, test = get_data(keyword='test')#,num=64*rows test_normal,test_anormal
print("normal size:{},anomaly size:{}".format(t_nor,t_anor))

# zeros_label = np.zeros(t_nor) #Label 0 means normal,size 1*BATCH
# zeros = zeros_label.T
# ones_label = np.ones(t_anor) #Label 1 means anormal,size 1*BATCH
test = np.expand_dims(test, axis=1)

Test_data = Variable(torch.from_numpy(test).float())
# Test_label = Variable(torch.from_numpy(Label).float())

Dnet = Discriminator()#.cuda(GPU_NUM)
# Dnet.load_state_dict(torch.load(Net_PATH))
Dnet = torch.load(Net_PATH)

#################################
T = Test_data
print(np.shape(T))
Results = Dnet(T)
Results = Results.data.numpy()
print(np.shape(Results))
# print(Results[988:1100,0])
failure_rate = 0#真当假
false_positive = 0#假当正
crect = 0

for i in range(t_nor):
    if Results[i,0] < 0.5:
        crect = crect + 1
    else:
        failure_rate += 1
for i in range(t_anor):
    if Results[i+t_nor,0] >= 0.5:
        crect = crect + 1
    else:
        false_positive += 1

t2 = time.time()
print('time test spent :',t2-t1)
print("crect number: {},total Accuracy rate:{}".format(crect,crect/(t_nor+t_anor)))
print('false detected rate respectively,normal:{},anomaly:{}'.format(failure_rate/t_nor,false_positive/t_anor))

# generate the segment data for filtering step
# results = np.random.randn(1,1,400)
# for i in range(16):
#     gen_data = Gnet(T[i*735 : (i+1)*735].cuda(GPU_NUM)).cpu()
#     # print(np.shape(gen_data))
#     results = np.vstack((results, gen_data.data.numpy()))
#
# results_3000 = results[1:].reshape(21,224000)
# print(np.shape(results_3000), np.shape(Test_label.data.numpy()))

# scio.savemat('dataset/QT_data/QT_GEN_add_klg_train400/add_klg_4channel_Gen_test_2500epoch.mat',{'data':results_3000})
# print(np.shape(Label))
# scio.savemat('dataset/QT_data/QT_GEN_add_klg_train400/Gen_test_label_3000.mat',{'data':Label})
# ##################################

# gen_data = Gnet(T.cuda(GPU_NUM)).cpu()
#
# wave = Test_data.data.numpy()
# results = gen_data.data.numpy()
# label = Test_label.data.numpy()
#
# # wave = Train_data.data.numpy()
# # results = gen_data.data.numpy()
# # label = Train_label.data.numpy()
#
# # change the length
# # wave = wave.reshape(-1,1000)
# # results = results.reshape(-1,1000)
# # label = label.reshape(-1,1000)
#
# print(np.shape(results))
# print(np.shape(wave))
# # plot data wave
# fig = plt.figure()
# c = 0
# for x in range(100):
#     ax1 = plt.subplot2grid((3, 1), (0, 0))
#     ax2 = plt.subplot2grid((3, 1), (1, 0))
#     ax3 = plt.subplot2grid((3, 1), (2, 0))
#
#     ax1.plot(wave[x+300,0], linewidth=0.4, label='ECG')
#     ax1.legend(loc='upper right')
#
#     ax2.plot(results[x+300, 0], linewidth=0.4, label='generated')
#     ax2.legend(loc='upper right')
#
#     ax3.plot(label[x+300], linewidth=0.4, label='label')
#     ax3.legend(loc='upper right')
#     ax3.set_ylabel('Amplitude')
#
#     plt.xlabel('Samples')
#
#     if not os.path.exists(Predictions):
#         os.makedirs(Predictions)
#
#     plt.savefig(Predictions + '_{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
#     c += 1
#     plt.close(fig)
#     # plt.show()

