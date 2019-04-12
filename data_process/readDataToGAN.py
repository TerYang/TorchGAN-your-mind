import pandas as pd
import numpy as np
import os
import threading as td
from queue import Queue
import multiprocessing as mp

# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/"
source_addr = "/home/gjj/PycharmProjects/ADA/netsData/hackingData/GANdata/"
# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/instrusion-dataset/test_data/data/"

def minbatch_test():

    file = 'Attack_free_dataset2_ID_Normalize.txt'
    url = os.path.join(source_addr, file)
    data1 = pd.read_csv(url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=64*64*100)
    data1 = data1.values.astype(np.float32)#
    # print(data1.shape)
    data1 = np.reshape(data1, (-1, 64, 22))
    print('normal :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    num1 = data1.shape[0]
    return num1,data1

def get_data():#train_normal,train_anormal,,num=64*10000

    files = os.listdir(source_addr)
    normals = []
    for file in files:
        normals.append( os.path.join(source_addr,file))
    # print('normals lenght:',normals)

    # normal0_name = os.path.basename(normals[0])
    # normal1_name = os.path.basename(normals[1])
    print('dataset:\n',files)
    # exit()
    data1 = pd.read_csv(normals[0], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data1 = data1.values.astype(np.float32)#

    # data2 = pd.read_csv(train_anormal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=num)
    data2 = pd.read_csv(normals[1], sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    data2 = data2.values.astype(np.float32)#,copy=True
    print('normal1 :ndim:{} dtype:{} shape:{}'.format(data1.ndim, data1.dtype, data1.shape))
    # print('finished:{}'.format(normal0_name))

    print('normal2 :ndim:{} dtype:{} shape:{}'.format(data2.ndim, data2.dtype, data2.shape))
    # print('finished:{}'.format(normal1_name))
    num1 = data1.shape[0]//64 #int(
    num2 =  data2.shape[0]//64

    data = np.concatenate((data1[:64*num1,:],data2[:64*num2,:]),axis=0)

    # data = np.reshape(data[:num*64,],(-1,64,22)).astype(np.float32)
    data = np.reshape(data, (-1, 64, 22))
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print("normal total has {}+{}={} blocks".format(num1,num2,num1+num2))
    print('done read files!!!\n')
    return num1+num2,data

# get_data()
minbatch_test()