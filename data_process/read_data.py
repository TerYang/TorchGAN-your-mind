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


def write(re_url,wri_url,skiprows=None,rows=None):
    name = re_url[65:]
    if skiprows == None:
        skiprows =0
    if rows == None:
        print('error:rows = None')
    print('name:{} skips:{} rows:{}'.format(name,skiprows,rows))
    # rows = rows*64
    # skiprows = skiprows * 64
    data = pd.read_csv(re_url, sep=None, dtype=np.str, header=None, engine='python',skiprows=skiprows*64,nrows=rows*64)
    data.to_csv(wri_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    print('{} done rows: {}'.format(name,data.index[-1]+1))

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


def collection_same_type_dat(source_addr,dire_addr=None,keyword='test'):#normalfiles,
    """#写测试数据合集
    anomalies:异常文件名集合
    source_addr：原始文件所在位置
    dire_addr：目的文件位置,默认在source_addr下面新建
    由于攻击重复了，impersonation 攻击有三个文件，只使用其中一个，attack free也只使用其中一个文件，另外一个备用
    """

    # total_data =   {'Attack_free_dataset2_ID_Normalize.txt':3713146,
    #                 'Fuzzy_attack_dataset_Normalize.txt':591990,
    #                 '170907_impersonation_Normalize.txt':659990,
    #                 'DoS_attack_dataset_Normalize.txt':656579,
    #                 '170907_impersonation_2_Normalize.txt':695365,
    #                 'Impersonation_attack_dataset_Normalize.txt':995472,
    #                 'Attack_free_dataset_Normalize.txt':2369398}
    total_data =   {#'Attack_free_dataset2_ID_Normalize.txt':3713146,
                    'Fuzzy_attack_dataset_Normalize.txt':591990,
                    '170907_impersonation_Normalize.txt':659990,
                    'DoS_attack_dataset_Normalize.txt':656579,
                    #'170907_impersonation_2_Normalize.txt':695365,
                    #'Impersonation_attack_dataset_Normalize.txt':995472,
                    'Attack_free_dataset_Normalize.txt':2369398}
    if keyword == 'test':
        weight = 0.2
    else:
        weight = 0.8


    normalfiles = ['Attack_free_dataset_Normalize.txt']


    anormal_urls = []#异常数据文件url 列表
    normal_urls = []#正常数据文件url 列表

    # addrs = os.listdir(source_addr)

    if dire_addr == None:
        dire_addr = source_addr + 'second_merge/'+ keyword +'/'
    if not os.path.exists(dire_addr):
        os.makedirs(dire_addr)

    addrs = list(total_data.keys())
    for addr in addrs:
        if addr not in normalfiles:
            anormal_urls.append(source_addr + addr)
        else:
            normal_urls.append(source_addr + addr)

    # print(anormal_urls,normal_urls)
    # exit()
    # 写数据normal
    wri_normal = dire_addr +keyword+'_normal.txt'
    pool_normal = mp.Pool(processes=len(normal_urls))
    for addr in normal_urls:
        pool_normal.apply(write,(addr,wri_normal,10000,4000,))#读取每一个正常数据集的前10000×64条数据，训练数据集
        # data = pd.read_csv(addr, sep=None, dtype=np.str, header=None, engine='python')
        # count1 += data.index[-1]+1
        # data.to_csv(normal_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    pool_normal.close()
    pool_normal.join()

    # #写训练中异常数据合集
    wri_anormal= dire_addr + keyword+'_anormal.txt'
    # count2 = 0
    pool_anoma = mp.Pool(processes=len(anormal_urls))


    for addr in anormal_urls:

        pool_anoma.apply(write, (addr, wri_anormal,4000,1000,))#读取每一个异常数据集的前4000×64条数据，训练数据集
        # data = pd.read_csv(read_url, sep=None, dtype=np.str, header=None, engine='python')
        # data.to_csv(write_url, sep=' ', index=False, header=False, mode='a', float_format='%.3f')
    pool_anoma.close()
    pool_anoma.join()

if __name__ == '__main__':
    # source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/"
    # test_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/test_data/"
    # train_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/train_data/"
    # merge_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/merge_data/"
    # # read_one = source_addr +'Attack_free_dataset2_ID_Normalize.txt'
    # # read_index(read_one)
    #
    #
    #
    #
    # if not os.path.exists(test_addr):  # 如果保存模型参数的文件夹不存在则创建
    #     os.makedirs(test_addr)
    # if not os.path.exists(train_addr):
    #     os.makedirs(train_addr)
    # if not os.path.exists(merge_addr):
    #     os.makedirs(merge_addr)
    #
    # read_url = []
    # test_url = []
    # train_url = []
    # num = []
    #
    # total_data =   {'Attack_free_dataset2_ID_Normalize.txt':3713146,
    #                 'Fuzzy_attack_dataset_Normalize.txt':591990,
    #                 '170907_impersonation_Normalize.txt':659990,
    #                 'DoS_attack_dataset_Normalize.txt':656579,
    #                 '170907_impersonation_2_Normalize.txt':695365,
    #                 'Impersonation_attack_dataset_Normalize.txt':995472,
    #                 'Attack_free_dataset_Normalize.txt':2369398}
    # normal_filename = ['Attack_free_dataset2_ID_Normalize.txt', 'Attack_free_dataset_Normalize.txt']
    #
    # for addr in addrs:
    #     read_url.append(source_addr + addr)
    #     test_url.append(test_addr + addr)
    #     train_url.append(train_addr+addr)
    #     num.append(total_data[addr])

    """获取文件大小，划分train和test数据集"""
    # pool = mp.Pool(processes=len(read_url))
    # # pool.map(gene_test_and_train_data,zip(read_url,test_url,train_url,num),)
    # pool.map(read_index,train_url,)
    # pool.close()
    # pool.join()
    # print("\nall had finished!!")

    """生成两个数据矩阵，normal,anormal"""
    # normal_url = []
    # anormal_url = []
    #
    # test_addrs = os.listdir(test_addr)
    # train_addrs =  os.listdir(train_addr)



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

    """多线程写训练数据"""
    # normal_filename = ['Attack_free_dataset2_ID_Normalize.txt', 'Attack_free_dataset_Normalize.txt']
    source_addr = "/home/gjj/PycharmProjects/ADA/ID-TIME_data/Batch_delNone_toNumpy/"

    # collection_same_type_dat(source_addr, keyword='train',)#dire_addr=None,
    collection_same_type_dat(source_addr, dire_addr=None, keyword='test',)#

    # print('normal:{} anormal:{}'.format(count1,count2))
    # read_index(w_test_normal)
    # read_index(w_test_anormal)
