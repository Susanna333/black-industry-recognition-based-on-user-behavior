#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import featureutils2 as futil
import gc

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
sns.set(style = 'white', context = 'notebook', palette = 'deep')
sns.set_style('white')
le = LabelEncoder()
dataroot = "./data/"

path1 = './data-firstround'
path2 = './data-secondround'

train_oper = pd.read_csv(path1 + '/operation_train_new.csv')
train_transac = pd.read_csv(path1 + '/transaction_train_new.csv')
test_oper_r1 = pd.read_csv(path1 + '/operation_round1_new.csv')
test_transac_r1 = pd.read_csv(path1 + '/transaction_round1_new.csv')

test_oper_r2 = pd.read_csv(path2 + '/test_operation_round2.csv')
test_transac_r2 = pd.read_csv(path2 + '/test_transaction_round2.csv')
trainTag = pd.read_csv(path1 + '/tag_train_new.csv')
submission = pd.read_csv(path2 + '/submit_example.csv')

train_oper['day_new'] = train_oper['day'].values
train_transac['day_new'] = train_transac['day'].values
test_oper_r1['day_new'] = test_oper_r1['day'].values+31
test_transac_r1['day_new'] = test_transac_r1['day'].values+31
test_oper_r2['day_new'] = test_oper_r2['day'].values+61
test_transac_r2['day_new'] = test_transac_r2['day'].values+61

use_oper = pd.concat([train_oper, test_oper_r1,test_oper_r2])
use_tran = pd.concat([train_transac, test_transac_r1, test_transac_r2])

###prepare
tmp_oper = use_oper.copy()
tmp_oper["is_oper"] = 1
use_all = pd.concat([tmp_oper,use_tran ])
use_all["is_oper"] = use_all["is_oper"].fillna(0)
use_all["ip"] = use_all["ip1"].fillna(use_all["ip2"])
del tmp_oper

dataset = [use_all,use_oper]
for data in dataset:
    data = futil.diffWindow600Secd(data, False)
    data['hour'] = data.datadate.map(futil.get_hour)
    data['ip'] = data['ip1'].fillna(data['ip2'])
use_all["device_code"] = use_all["device_code1"].fillna(use_all["device_code2"])
use_all["device_code"] = use_all["device_code"].fillna(use_all["device_code3"])
use_tran["device_code"] = use_tran["device_code1"].fillna(use_tran["device_code2"])
use_tran["device_code"] = use_tran["device_code"].fillna(use_tran["device_code3"])
use_oper.wifi = use_oper.wifi.fillna("Phonetraffic")
use_tran = futil.diffWindow600Secd(use_tran, True)
use_tran["hour"] = use_tran.datadate.map(futil.get_hour)

for col in ['bal','trans_amt']:
    use_tran[col+'_increase'] = use_tran[col+'_dif'].map(lambda x: x if x >0 else 0)
    use_tran[col+'_decrease'] = use_tran[col+'_dif'].map(lambda x: x if x <0 else 0)
    use_tran[col+'_difabs'] = use_tran[col+'_dif'].map(abs)
use_tran['bal_decrease2amt'] = use_tran['bal_decrease']/use_tran['trans_amt']
use_tran['bal_decrease2amt'] = use_tran['bal_decrease2amt'].map(abs)

for col in ["os","wifi"]:
    use_all[col] = use_all.groupby("UID")[col].apply(lambda x: x.ffill())

for col in ['version','mode']:
    use_all[col + "_before_tran"] = use_all.groupby("UID")[col].apply(lambda x: x.ffill()) 
    use_all[col + "_after_tran"] = use_all.groupby("UID")[col].apply(lambda x: x.bfill()) 
    

#####block1#####

data = pd.read_csv('./data/data.csv')
block1 = pd.DataFrame()
block1["UID"] = data["UID"]

##短时间内不同客户端版本号数量
gp1 = use_all.groupby(["UID",'time_group'])['version'].agg({ "version_short_time":"nunique" }).reset_index()
gp = gp1.groupby("UID")["version_short_time"].agg({ "version_short_time_max":"max",
                                                  "version_short_time_mean":"mean"}).reset_index()
block1 = block1.merge(gp,on = ["UID"],how = "left")
del gp,gp1

##操作类型
gp1 = use_all[use_all.device2.isnull()].groupby("UID")["mode"].agg({
                                "device2null_most_mode":lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan ,
                                "device2null_most_mode_cnt":lambda x:x.value_counts().values[0] if len(x.value_counts())>1 else np.nan ,
                                "device2null_scnd_mode":lambda x: x.value_counts().index[1] if len(x.value_counts())>1 else np.nan ,
                                "device2null_scnd_mode_cnt":lambda x:x.value_counts().values[1] if len(x.value_counts())>1 else np.nan 
                               }).reset_index()
gp1["device2null_scnd2most_mode_ratio"] = gp1["device2null_scnd_mode_cnt"]/gp1["device2null_most_mode_cnt"]
to_encode = list( set(gp1.device2null_most_mode.map(str).unique().tolist() + gp1.device2null_scnd_mode.map(str).unique().tolist()) )
le.fit(to_encode)
gp1.device2null_most_mode = le.transform(gp1.device2null_most_mode.map(str))
gp1.device2null_scnd_mode = le.transform(gp1.device2null_scnd_mode.map(str))
block1 = block1.merge(gp1,on = ["UID"],how = "left")
del gp1


######block2######
# device2/geo_code/  os/version/success/wifi/  channel/day/hour/time_diff_secd
# count version_cnt in time_froup 表征刷机（版本更替频繁 短时间） 最好 对version 做 befeore after操作
# 避免刷机影响 求 max version in time_group 后再求最小版本
# 刷机pattern 2 一组时间内，要么交易，要么操作 deveice2 为空
# 统计 mode对 device2 各种操作缺失，或者缺失下的 mode 第一是什么第二是什么 第二/第一 比例
# 统计device_code1空值比
# device2 Tag1 的 IPTHON 5S很多，0的基本没有5S，直接对用户手机device2 most值lable encode
block2 = pd.DataFrame()
block2["UID"] = block1["UID"]

filename1 = './group_aggregations.txt'
with open(filename1,'r') as f:
    group_aggregations1 = f.read()
group_aggregations1 = eval(group_aggregations1)


### block3
block3 = pd.DataFrame()
block3["UID"] = block1["UID"]

group_aggregations2 = [
    # Variance in day,hour,ip,channel,merchant,time_group, conditon time_diff_secd/day
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},


{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},


{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},
{'groupby': ['UID','day'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'},

{'groupby': ['UID','channel'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'}]


def groupfunction(group_aggregations,data,block):
    for spec in group_aggregations:
        celldata = pd.DataFrame()
        celldata["UID"] = data["UID"]

        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

        # Info
        print("Grouping by {}, and aggregating {} with {}".format(spec['groupby'], spec['select'], agg_name))

        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # Perform the groupby
        if "query" in spec:
            d_gp = data.query(spec['query'])[all_features]
        else:
            d_gp = data[all_features]
        gp = d_gp.groupby(spec['groupby'])[spec['select']].agg(spec['agg']).reset_index().                                        rename(index=str, columns={spec['select']: new_feature})
        gpc = gp.groupby('UID')[new_feature].agg({new_feature + "_max" :"max",
                                                  new_feature + "_min" :"min",
                                                  new_feature + "_mean" :"mean"}).reset_index()

        block = block.merge(gpc,on = "UID" ,how = 'left')

        del gp,gpc,d_gp
        gc.collect()
    return block
    
    
block2 = groupfunction(group_aggregations1,use_all,block2)
block3 = groupfunction(group_aggregations2,use_tran,block3)

######## block4 ########
# 多角度下用户属性扩散后的指标，比如用户多个ip中用户量/交易量/地址数各是多少，用户在维度可统计的（交易量）占据属性本身比例
# 属性可算什么/用户是否同时可算，比率还是比值，属性下用户可算值均值，用户在属性可算值比去这个值，或者统计所有属性值的该值（可算值分布，不是属性值分布）的分布
#                |---ip1/mac1  num of users/trans/geos and USR ratio in ip1
# USR(summary) --|---ip2/mac2  num of users/trans/geos and USR ratio in ip2
#                |---ip3/mac3  num of users/trans/geos and USR ratio in ip3
# is_ratio 统计用户 可算值/属性可算值均值 可算值/属性可算值总值

block4 = pd.DataFrame()
block4["UID"] = block1["UID"]

dataUse = {"use_tran":use_tran,"use_oper":use_oper}
GroupOnCags = [
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["mac1"    ,'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["geo_code",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["device_code",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["device_code1",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_oper" ,'is_ratio':{'gpby':["mac2",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["ip1",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'],'use':"use_tran" ,'is_ratio':{'gpby':["merchant",'UID'],"slt":'day','agn':'trans','ag':"count"} }]


for spec in GroupOnCags:
    celldata = pd.DataFrame()
    celldata["UID"] = data["UID"]       
    is_ratio = spec['is_ratio']
    agg_n = is_ratio['agn'] if 'agn' in is_ratio else is_ratio['ag']
    ratio_on_name = '{}_{}'.format('_'.join(is_ratio['gpby']), agg_n)
    
    print("Grouping by {}, and aggregating {} with {}".format(is_ratio['gpby'], is_ratio['slt'], agg_n))

    each = dataUse[spec['use']].groupby(is_ratio['gpby'])[is_ratio['slt']].agg(is_ratio['ag']).reset_index().                                                                rename(index=str, columns={is_ratio['slt']: ratio_on_name})
#             each = each.fillna(0.00001)
    total = each.groupby(is_ratio['gpby'][0])[ratio_on_name].agg({ ratio_on_name + "_mean":"mean",
                                                                   ratio_on_name + "_sum":"sum"}).reset_index()
    gp = pd.merge(each,total,on=[ is_ratio['gpby'][0]] )
    gp[ "user_ratio_on_" + ratio_on_name + "_mean" ] = gp[ratio_on_name]/gp[ratio_on_name + "_mean"]
    gp[ "user_ratio_on_" + ratio_on_name + "_sum" ] = gp[ratio_on_name]/gp[ratio_on_name + "_sum"]
    gp = gp.drop(ratio_on_name,axis =1 )
    for c in gp.columns:
        if c in is_ratio['gpby']:
            pass
        else:
            gpc = gp.groupby("UID")[c].agg({ c + "_maxon" +is_ratio['gpby'][0] :"max" ,
                                             c + "_avgon" +is_ratio['gpby'][0] :"mean" ,
                                             c + "_minon" +is_ratio['gpby'][0] :"min"}).reset_index()
            block4 = block4.merge(gpc,on = "UID" ,how = 'left')
            del gpc

    del gp,each,total
    gc.collect()

    
for bk in [block2,block3,block4]:
    block1 = block1.merge(bk,on=['UID'],how='left')
block1.to_csv(dataroot  + 'groupdata.csv', index = False)