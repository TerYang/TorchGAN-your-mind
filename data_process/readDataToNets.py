import pandas as pd
import numpy as np
import os
import threading as td
from queue import Queue
import multiprocessing as mp

# source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/"
source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/instrusion-dataset/Batch_delNone_toNumpy/second_merge/"
# test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/second_merge/test/"

# def job(url,q):
#     min_bounder = 0
#     data1 = []
#     data = pd.read_csv(url, sep=None, header=None,dtype=np.float32, engine='python',encoding='utf-8')
#     round = int((data.index[-1]+1)/64)#数据的条数
#     print("{} data number is:{} split to {} blcoks\n".format(url[60:],data.index[-1],round))
#
#     for i in range(1,round+1,1):
#
#         if i == 1:
#             print('data.loc[min_bounder:i*63,:].values type:',type(data.iloc[min_bounder:i*63,:].values))
#             print(type(data.iloc[min_bounder:i*63,:].values))
#
#             data1= np.array(data.iloc[min_bounder:i*63,:].values).reshape((64,22))
#
#             # data1 = np.expand_dims(data1,axis=0)
#         else:
#             print('data1 type:{},shape:{}'.format(type(data1),data1.shape))
#             # data1 = np.concatenate((data1,np.array(data.loc[min_bounder:i*63,:].values)),axis=0)
#             data1 = np.append(data1,np.array(data.iloc[min_bounder:i*63,:].values),axis=0)
#
#         min_bounder = i*63
#         if i%100==0:
#             print('loop:{}'.format(i))
#     # data1 = np.array(data1).reshape(round,64,22)
#     data1 = np.reshape(data1,(-1,64,22))
#     q.put(data1)


# def job(url,q):
#     min_bounder = 0
#     data1 = []
#     data = pd.read_csv(url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',chunksize=64)
#     # print(url)
#     # exit()
#     name = url[-12:]
#     for i,o1 in enumerate(data):
#         size = o1.index[-1]
#         # print(o1)
#         # exit()
#         # print("{} data number is:{} split to {} blcoks\n".format(url[60:],size,i))
#         if size < 63:
#             print(size,i)
#             count = i
#             continue
#         else:
#             if i == 0:
#                 data1 = o1.values.reshape(64, 22).astype(np.float32)
#             if i != 0:
#                 try:
#                     data1 = np.concatenate((data1,o1.values.reshape(64, 22).astype(np.float32)),axis=0)
#                 except ValueError:
#                     print("file name:{},loop:{} {}".format(name,i,'-'*50))
#                 if i%500==0:
#                     print('loop:{}'.format(i))
#                     print("data1 shape:{} data1 type:{}".format(data1.shape,type(data1)))
#     # data1 = np.reshape(data1,(-1,64,22))
#     q.put(data1)



# def job(url,q):
#     min_bounder = 0
#     # data1 = []
#     data = pd.read_csv(url, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=640000)
#     # print(url)
#     # exit()
#     # name = url[-12:]
#     # for i,o1 in enumerate(data):
#     #     size = o1.index[-1]
#     #     # print(o1)
#     #     # exit()
#     #     # print("{} data number is:{} split to {} blcoks\n".format(url[60:],size,i))
#     #     if size < 63:
#     #         print(size,i)
#     #         count = i
#     #         continue
#     #     else:
#     #         if i == 0:
#     #             data1 = o1.values.reshape(64, 22).astype(np.float32)
#     #         if i != 0:
#     #             try:
#     #                 data1 = np.concatenate((data1,o1.values.reshape(64, 22).astype(np.float32)),axis=0)
#     #             except ValueError:
#     #                 print("file name:{},loop:{} {}".format(name,i,'-'*50))
#     #             if i%500==0:
#     #                 print('loop:{}'.format(i))
#     #                 print("data1 shape:{} data1 type:{}".format(data1.shape,type(data1)))
#     # data1 = np.reshape(data1,(-1,64,22))
#     data1 = data.values.astype(np.float32)
#     print('had read file____________________')
#     # return data1
#     q.put(data1)

def get_data(keyword='train'):#train_normal,train_anormal,,num=64*10000

    normal = source_addr + keyword + '/' + keyword+'_normal.txt'
    anomaly = source_addr +  keyword + '/' + keyword+'_anormal.txt'
    # print(normal)
    # exit()
    # data1 = pd.read_csv(train_normal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=num)
    data1 = pd.read_csv(normal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    normal_data = data1.values.astype(np.float32)#

    # data2 = pd.read_csv(train_anormal, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8',nrows=num)
    data2 = pd.read_csv(anomaly, sep=None, header=None,dtype=np.str, engine='python',encoding='utf-8')
    anormal_data = data2.values.astype(np.float32)#,copy=True

    print('anormal_data :ndim:{} dtype:{} shape:{}'.format(normal_data.ndim,normal_data.dtype,normal_data.shape))

    print('finished:{}'.format(normal[60:]))
    print('anormal_data :ndim:{} dtype:{} shape:{}'.format(anormal_data.ndim,anormal_data.dtype,anormal_data.shape))
    print('finished:{}'.format(anomaly[60:]))
    num_anor = int(anormal_data.shape[0]/64)
    num_nor =  int(normal_data.shape[0]/64)

    print(num_anor,num_nor)

    data = np.concatenate((normal_data,anormal_data),axis=0)
    data = np.reshape(data,(-1,64,22))
    print('data :ndim:{} dtype:{} shape:{}'.format(data.ndim, data.dtype, data.shape))
    print('done read files!!!\n')
    return num_nor,num_anor,data.astype(np.float32)


# get_data()