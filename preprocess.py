#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import datetime
import time
from dateutil.parser import parse

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
sns.set(style = 'white', context = 'notebook', palette = 'deep')
sns.set_style('white')

# 读取数据
path1 = './data-firstround'
op_train = pd.read_csv(path1 + '/operation_train_new.csv')
tr_train = pd.read_csv(path1 + '/transaction_train_new.csv')
tag_train = pd.read_csv(path1 + '/tag_train_new.csv')
op_test1 = pd.read_csv(path1 + '/operation_round1_new.csv')
tr_test1 = pd.read_csv(path1 + '/transaction_round1_new.csv')

path2 = './data-secondround'
op_test2 = pd.read_csv(path2 + '/test_operation_round2.csv')
tr_test2 = pd.read_csv(path2 + '/test_transaction_round2.csv')

path3 = './geocode'
geo_hash = pd.read_csv(path3 + '/GeoDetails.csv')
del geo_hash['street'],geo_hash['street_number']

# 转换时间天数，整体的时间顺序为： 训练集时间 早于 初赛测试集时间 早于 复赛测试集时间
op_train['day_new'] = op_train['day'].values
tr_train['day_new'] = tr_train['day'].values
op_test1['day_new'] = op_test1['day'].values+30
tr_test1['day_new'] = tr_test1['day'].values+30
op_test2['day_new'] = op_test2['day'].values+61
tr_test2['day_new'] = tr_test2['day'].values+61

# 读取手机品牌字典
with open('brand_dict.txt','r') as f:
    brand_dict = f.read()
brand_dict = eval(brand_dict)

# 合并
op_data = pd.concat([op_train,op_test1,op_test2], axis=0, ignore_index=True)
tr_data = pd.concat([tr_train,tr_test1,tr_test2], axis=0, ignore_index=True)


def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + datetime.timedelta(days = days) # parse 统一转换日期格式
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

def timestamp_transform(a):
    #将python的datetime转换为unix时间戳
    timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
    timeStamp = float(time.mktime(timeArray))
    #将unix时间戳转换为python的datetime
    return timeStamp

def time_transfer(data,basetime):
    data['actionTime'] = data['day_new'].apply(lambda x: date_add_days(basetime, x))
    data['actionTime'] = data['actionTime'].map(str)+' '+data['time'].map(str)
    data['actionTimestamp'] = data['actionTime'].apply(lambda x: timestamp_transform(x))
    return data

def clear_device_code(code1, code2, code3):
    if pd.isna(code1):
        code1 = ''
    if pd.isna(code2):
        code2 = ''
    if pd.isna(code3):
        code3 = ''
    return code1+code2+code3

def definite_device_code(data):
    data['device_code'] = data.apply(lambda x: clear_device_code(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
    data['device_code'].replace({'':np.nan}, inplace=True)
    return data

# 1:ios 2:android 3:unknown
def check_device_type(code1, code2, code3):
    if (pd.isna(code1)==False)|(pd.isna(code2)==False):
        return 1
    if pd.isna(code3)==False:
        return 2
    else:
        return -1

def device2_clear(x):
    x = str(x)
    y = ''
    if x == 'nan':
        y = '-1'
    else:
        for keys, items in brand_dict.items():
            for i in items:
                if x.find(i) != -1:
                    y = keys
        if len(y) == 0:
            y = 'others'
    return y

# 重新定义ip
def ip_clear(ip1, ip2):
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==False):
        return ip2
    if (pd.isna(ip1)==False)&(pd.isna(ip2)==True):
        return ip1
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==True):
        return np.nan
    else:
        return ip1

def is_china(x):
    if x=='中国':
        return 1
    if pd.isna(x)==True:
        return -1
    else:
        return 0

# 1:computer 2:phone 3:both
def check_oper_env(ip1, ip2):
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==False):
        return 1
    if (pd.isna(ip1)==False)&(pd.isna(ip2)==True):
        return 2
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==True):
        return -1
    else:
        return 3


def preprocess(dataset):
    for data in dataset:
        # 定义完整时间
        data = time_transfer(data,'2018-08-31')
        data['device_type'] = data.apply(lambda x: check_device_type(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
        # 重新定义device_code
        data = definite_device_code(data)
        # 判断苹果/安卓机
        data['device_type'] = data.apply(lambda x: check_device_type(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
        del data['device_code1'], data['device_code2'], data['device_code3']
        # 提取出手机品牌
        data['device_brand'] = data['device2'].apply(lambda x: device2_clear(x))
        # 匹配地理信息
        data = data.merge(geo_hash, on=['geo_code'], how='left')
        # 时间信息
        data['hour'] = data['time'].str[0:2].astype(int)
        # 是否中国地区
        data['is_china'] = data['nation'].apply(lambda x: is_china(x))
    return dataset



def preprocess2(data):
    # 设备版本信息
    data['os_version'] = data['os'].map(str)+'_'+data['version'].map(str)
    # 是否是wifi环境
    data['is_wifi_env'] = data['wifi'].apply(lambda x: 1 if pd.isna(x)==False else 0)
    # 是否电脑
    data['oper_ip_env'] = data.apply(lambda x: check_oper_env(x['ip1'], x['ip2']), axis=1)
    data['ip'] = data.apply(lambda x: ip_clear(x['ip1'], x['ip2']), axis=1)
    data['ip_sub'] = data.apply(lambda x: ip_clear(x['ip1_sub'], x['ip2_sub']), axis=1)
    del data['ip1'], data['ip1_sub'], data['ip2'], data['ip2_sub']
    return data

dataset = [op_data, tr_data]
op_data,tr_data = preprocess(dataset)
op_data = preprocess2(op_data)
tr_data = tr_data.rename(columns={'ip1': 'ip','ip1_sub': 'ip_sub'})

# 设备信息缺失程度
device_cols1 = ['os', 'version', 'device1', 'device2', 'device_code', 'mac1']
device_cols2 = ['device1', 'device2', 'device_code', 'mac1']
op_data['device_miss_cnt'] = op_data[device_cols1].isnull().sum(axis=1)
tr_data['device_miss_cnt'] = tr_data[device_cols2].isnull().sum(axis=1)
# 环境信息缺失程度
env_cols1 = ['wifi', 'ip', 'ip_sub', 'mac2', 'geo_code']
env_cols2 = ['ip', 'ip_sub', 'geo_code']
op_data['env_miss_cnt'] = op_data[env_cols1].isnull().sum(axis=1)
tr_data['env_miss_cnt'] = tr_data[env_cols2].isnull().sum(axis=1)
# # 缺失值填补
op_data['success'].fillna(-1, inplace=True)
del op_data['day_new'], tr_data['day_new']


op_data.to_csv('./data/operation_pre.csv', encoding='gbk', index=False)
tr_data.to_csv('./data/transaction_pre.csv', encoding='gbk', index=False)

