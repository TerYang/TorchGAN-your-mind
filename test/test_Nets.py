import torch
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ECG_exp02.data_process import read_data , restore_data , plot_train

# Hyper parameters
TEST_dim = 198
INPUT_dim = 310
DATA_num = 30
GPU_NUM = 2

SAVE_NET_PATH_generator = '/root/netdisk/1Working/NN_saved/GAN_G_EM_revise_addSigmoid_20190114_QRS_310.pkl'
# SAVE_NET_PATH_generator = '/root/netdisk/1Working/NN_saved/GAN_G_MA_20180628_QRS_310_12.pkl'
# SAVE_NET_PATH_second = '/root/netdisk/1Working/NN_saved/net_newdata_SecondTrain.pkl'
# NUM_DATA = ['103','105','111','116','122','205','213','219','223','230']
NUM_DATA = np.arange(DATA_num)
RMSE = np.random.randn(DATA_num)
SNR = np.random.randn(DATA_num)
Allsnr = np.random.randn(TEST_dim * DATA_num)   #
#read data
_ , __, testD, testL , MAX , MIN  = read_data()

print(np.shape(testD),np.shape(testL))

def data_trans():     # transform the data to torch & Variable
    TestdataM = torch.from_numpy(testD).float()
    TestLabelM = torch.from_numpy(testL).float()
    return Variable(TestdataM).cuda(GPU_NUM),Variable(TestLabelM)

def cal_rmse(Data,Label):     # calculate root MSE
    # Data = Data.data.numpy()
    # Label = Label.data.numpy()

    Temp = (Data - Label) ** 2
    for n in range(DATA_num):
        RMSE[n] = np.sqrt(np.mean(Temp[n * TEST_dim : (n + 1) * TEST_dim]))  # RMSE
    return RMSE

def cal_snr(Data,Label):      # calculate SNR
    # Data = Data.data.numpy()
    # Label = Label.data.numpy()

    Data = Data.reshape((DATA_num,TEST_dim*INPUT_dim))  # shape of 30*80000
    Label = Label.reshape((DATA_num,TEST_dim*INPUT_dim))

    Noise = (Data - Label) ** 2
    Label = Label ** 2
    Ntemp = np.sum(Noise, axis=1)
    Ltemp = np.sum(Label, axis=1)

    for i in range(DATA_num):
        SNR[i] = 10 * np.log10(Ltemp[i]/Ntemp[i])

    return SNR

def de_normalize(NoiseDATA,DenoisedDATA,LabelDATA):     # prepare the data to plot ECG wave
    Noise = NoiseDATA.data.numpy() #* (MAX - MIN) + MIN
    DeNoised = DenoisedDATA.data.numpy() #* (MAX - MIN) + MIN
    Original = LabelDATA.data.numpy() #* (MAX - MIN) + MIN

    return Noise , DeNoised , Original

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.sigmoid(self.hidden1(x))     # activation function for hidden layer
        x = F.sigmoid(self.hidden2(x))     # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

# transform data
Test , Label = data_trans()
#### plot noise data
# noise_data = np.copy(testD[:30])

net_generator = torch.load(SAVE_NET_PATH_generator)     # load generator net
# net_second = torch.load(SAVE_NET_PATH_second)     # load second denoised net
net_generator = net_generator.cpu()
net_generator = net_generator.cuda(GPU_NUM)

denoised_data = net_generator(Test).cpu() # shape( -1,300) , maybe 8000*300
# denoised_data = net_second(denoised_G).cpu() # shape( -1,300) , maybe 8000*300

# rmse = cal_rmse(denoised_data,Label)    # calculate rmse
# snr = cal_snr(denoised_data,Label)      # calculate snr

Noise , DeNoised , Original = de_normalize(Test.cpu(),denoised_data,Label)  # de  normalization
rmse = cal_rmse(DeNoised,Original)    # calculate rmse
snr = cal_snr(DeNoised,Original)      # calculate snr

# rmse = cal_rmse(Noise,Original)    # calculate rmse
# snr = cal_snr(Noise,Original)      # calculate snr

# plot bar snr
# plt.bar(NUM_DATA,snr,facecolor='#9999ff', edgecolor='white')
# plt.show()

print('rmse :',rmse)
print('snr :',snr)
print('MEAN_SNR:',np.mean(snr),"MEAN_RMSE:",np.mean(rmse))
print('MAX,MIN',MAX,MIN)

noise_wave , de_wave , label = restore_data(Noise , DeNoised ,Original)

print(np.shape(noise_wave),np.shape(de_wave),np.shape(label))

# print('shape of noise_wave:\n',np.shape(noise_wave))
# print('value of noise_wave:\n',noise_wave[0][:1000])
# print('value of de_wave:\n',noise_wave[0][:1000])
# print('value of label:\n',noise_wave[0][:1000])

# scio.savemat('data_loss_function/noise.mat',{'noise':noise_wave[0]})
# scio.savemat('data_loss_function/d_GANloss.mat',{'d_GANloss':de_wave[0]})
# scio.savemat('data_loss_function/label.mat',{'label':label[0]})

plot_train(noise_wave,de_wave,label)

# # plt.savefig('./data_figures/newdata_em_1_25.png')