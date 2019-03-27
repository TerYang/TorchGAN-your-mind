import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import threading as td
import multiprocessing as mp
from queue import Queue
import time


def read_index(url):

    read_addr = url
    # print('\n',url)
    name = read_addr[53:]
    # print('\n',name)
    # return 0
    data = pd.read_csv(read_addr, sep=None, dtype=np.str, header=None, engine='python')
    print("'{}':{}".format(name, data.index[-1]+1))


def gene_test_and_train_data(url):

    read_addr = url[0]
    write_train_addr = url[1]
    write_test_addr = url[2]
    num = int(url[3]*0.8)
    print('{} start processing'.format(read_addr[65:]))

    data = pd.read_csv(read_addr, sep=None, dtype=np.str, header=None, engine='python')
    data.loc[:num,:].to_csv(write_train_addr, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    data.loc[num:,:].to_csv(write_test_addr, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    print('{} finished'.format(read_addr[65:]))


if __name__ == '__main__':
    source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/"
    test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/test_data/"
    train_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/train_data/"
    merge_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/merge_data/"
    # read_one = source_addr +'Attack_free_dataset2_ID_Normalize.txt'
    # read_index(read_one)


    addrs = os.listdir(source_addr)

    if not os.path.exists(test_addr):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(test_addr)
    if not os.path.exists(train_addr):
        os.makedirs(train_addr)
    if not os.path.exists(merge_addr):
        os.makedirs(merge_addr)

    read_url = []
    test_url = []
    train_url = []
    num = []

    total_data =   {'Attack_free_dataset2_ID_Normalize.txt':3713146,
                    'Fuzzy_attack_dataset_Normalize.txt':591990,
                    '170907_impersonation_Normalize.txt':659990,
                    'DoS_attack_dataset_Normalize.txt':656579,
                    '170907_impersonation_2_Normalize.txt':695365,
                    'Impersonation_attack_dataset_Normalize.txt':995472,
                    'Attack_free_dataset_Normalize.txt':2369398}
    # print(addrs)
    # exit()
    for addr in addrs:
        read_url.append(source_addr + addr)
        test_url.append(test_addr + addr)
        train_url.append(train_addr+addr)
        num.append(total_data[addr])

    """获取文件大小，划分train和test数据集"""
    # pool = mp.Pool(processes=len(read_url))
    # # pool.map(gene_test_and_train_data,zip(read_url,test_url,train_url,num),)
    # pool.map(read_index,train_url,)
    # pool.close()
    # pool.join()
    # print("\nall had finished!!")

    """生成两个数据矩阵，normal,anormal"""
    normal_url = []
    anormal_url = []

    test_addrs = os.listdir(test_addr)
    train_addrs =  os.listdir(train_addr)



    """#写训练中数据合集"""
    # normal_filename = ['Attack_free_dataset2_ID_Normalize.txt','Attack_free_dataset_Normalize.txt']
    # for addr in train_addrs:
    #     if addr not in normal_filename:
    #         anormal_url.append(train_addr + addr)
    #     else:
    #         normal_url.append(train_addr + addr)
    # print(anormal_url,'\n',normal_url)
    # train_normal_url = merge_addr + 'normal.txt'
    # count1 = 0
    # for addr in normal_url:
    #
    #     data = pd.read_csv(addr, sep=None, dtype=np.str, header=None, engine='python')
    #     count1 += data.index[-1]+1
    #     data.to_csv(train_normal_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    #
    # # #写训练中异常数据合集
    # train_anormal_url = merge_addr + 'anormal.txt'
    # count2 = 0
    # for addr in anormal_url:
    #     data = pd.read_csv(addr, sep=None, dtype=np.str, header=None, engine='python')
    #     count2 += data.index[-1] +1
    #     data.to_csv(train_anormal_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    """#写测试数据合集"""
    normal_filename = ['Attack_free_dataset2_ID_Normalize.txt','Attack_free_dataset_Normalize.txt']
    for addr in test_addrs:
        if addr not in normal_filename:
            anormal_url.append(test_addr + addr)
        else:
            normal_url.append(test_addr + addr)
    # 写数据
    test_normal_url = merge_addr + 'test_normal.txt'
    count1 = 0
    for addr in normal_url:
        data = pd.read_csv(addr, sep=None, dtype=np.str, header=None, engine='python')
        count1 += data.index[-1]+1
        data.to_csv(test_normal_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')

    # #写训练中异常数据合集
    test_anormal_url = merge_addr + 'test_anormal.txt'
    count2 = 0
    for addr in anormal_url:
        data = pd.read_csv(addr, sep=None, dtype=np.str, header=None, engine='python')
        count2 += data.index[-1] +1
        data.to_csv(test_anormal_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')

    print('normal:{} anormal:{}'.format(count1,count2))
    # read_index(w_test_normal)
    # read_index(w_test_anormal)
