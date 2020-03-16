#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import warnings
import featureutils as futl
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

operation_df = pd.read_csv('./data/operation_pre.csv', encoding='gbk')
transaction_df = pd.read_csv('./data/transaction_pre.csv', encoding='gbk')
tag_train = pd.read_csv('./data-firstround/tag_train_new.csv')
submission1 = pd.read_csv('./data-firstround/提交样例.csv')
submission2 = pd.read_csv('./data-secondround/submit_example.csv')
data = pd.concat([tag_train, submission2], axis=0, ignore_index=True)

###################################
#############特征工程##############
###################################

operation_df['action_type'] = 1
transaction_df['action_type'] = 2
transaction_df['success'] = 1

for temp in [operation_df,transaction_df]:
    temp['date'] = temp['actionTime'].str[:10].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

inner_cols = list(set(operation_df).intersection(set(transaction_df))) # 合并operation_df和transaction_df有共同的特征列
user_action = pd.concat([operation_df[inner_cols], transaction_df[inner_cols]], axis=0, ignore_index=True)
user_action = user_action.sort_values(by=['UID', 'actionTime'])
operation_df = operation_df.sort_values(by=['UID', 'actionTime'])
transaction_df = transaction_df.sort_values(by=['UID', 'actionTime'])

#### 计算用户总交易次数、操作+交易次数、交易次数占比 ####
gp = user_action.groupby(['UID', 'action_type']).size().unstack().reset_index().fillna(0)
gp.columns = ['UID', 'operation_cnt', 'trade_cnt']
gp['action_cnt'] = gp['operation_cnt'] + gp['trade_cnt']
gp['trade_ratio'] = gp['trade_cnt'] / gp['action_cnt']
gp['trade_operation_ratio'] = gp['trade_cnt'] / (0.01+gp['operation_cnt'])
data = data.merge(gp, on=['UID'], how='left').fillna(0)

#### 是否有操作或者交易 ####
data['has_operation'] = data['operation_cnt'].apply(lambda x: 1 if x>0 else 0)
data['has_trade'] = data['trade_cnt'].apply(lambda x: 1 if x>0 else 0)

#### 用户操作成功次数、失败次数、成功失败次数差值、成功/失败占比, 交易次数占比和成功/失败占比的比值 ####
stats = user_action[user_action['action_type']==1].copy()
gp = stats.groupby(['UID', 'success']).size().unstack().reset_index().fillna(0)
gp.columns = ['UID','unknown_operation_cnt', 'failure_operation_cnt', 'success_operation_cnt']
gp['operation_cnt_diff'] = gp['success_operation_cnt']-gp['failure_operation_cnt']
data = data.merge(gp, on=['UID'], how='left')
data['success_operation_ratio'] = data['success_operation_cnt'] / (0.01+data['operation_cnt'])
data['failure_operation_ratio'] = data['failure_operation_cnt'] / (0.01+data['operation_cnt'])
data['unknown_operation_ratio'] = data['unknown_operation_cnt'] / (0.01+data['operation_cnt'])
data['trade_success_operation_ratio'] = data['trade_ratio'] / (0.01+data['success_operation_ratio'])
data['trade_failure_operation_ratio'] = data['trade_ratio'] / (0.01+data['failure_operation_ratio'])
data['trade_unknown_operation_ratio'] = data['trade_ratio'] / (0.01+data['unknown_operation_ratio'])

##### 用户操作/交易/整体使用设备/环境种类个数/差值/重合度 ####
cols = ['device1', 'device2', 'device_code', 'mac1', 'device_brand','ip', 'ip_sub', 'geo_code', 'nation', 'city', 'district']
tmp1 = user_action[user_action['action_type']==1].copy()
tmp2 = user_action[user_action['action_type']==2].copy() 
    
for col in cols:
    gp = user_action.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'action_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')
    gp1 = tmp1.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'operation_'+col+'_nunique'})
    data = data.merge(gp1, on=['UID'], how='left')
    gp2 = tmp2.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'trade_'+col+'_nunique'})
    data = data.merge(gp2, on=['UID'], how='left')
    data['trade_operation_'+col+'_diff'] = data['trade_'+col+'_nunique'] - data['operation_'+col+'_nunique']
    
    gp3 = tmp1.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp4 = tmp2.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp3.columns = ['UID', 'operation_'+col+'_list']
    gp4.columns = ['UID', 'trade_'+col+'_list']
    data = data.merge(gp3, on=['UID'], how='left')
    data = data.merge(gp4, on=['UID'], how='left')
    data['trade_operation_'+col+'_intersection_ratio'] = data.apply(lambda x:                                 futl.calculate_intersection_ratio(x['operation_'+col+'_list'], x['trade_'+col+'_list']), axis=1)    
    del data['operation_'+col+'_list'], data['trade_'+col+'_list']
    
#### 用户各操作交易类型种类个数####
op_feature = ['device_type','mac2','mode','os_version','wifi']
tr_feature = ['acc_id1','acc_id2','acc_id3','amt_src1','amt_src2','channel','code1','code2','market_code',              'market_type','trans_type1','trans_type2']
for col in op_feature:
    gp = operation_df.groupby('UID')[col].nunique().reset_index().rename(columns={col: 'operation_'+col+'_nunique'})
for col in tr_feature:
    gp = transaction_df.groupby('UID')[col].nunique().reset_index().rename(columns={col: 'transaction_'+col+'_nunique'})

#### 设备/环境缺失个数平均值、最大值 ####
gp = user_action.groupby(['UID'])[['device_miss_cnt', 'env_miss_cnt']].agg({'mean', 'max'})
gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')

used_cols = ['ip_sub', 'device1', 'mac1','ip','device_code', 'device2', 'geo_code']
for col in used_cols:
    gp = user_action.groupby(['UID'])[col].count().reset_index().rename(columns={col: 'action_nonan_'+col+'_count'})
    data = data.merge(gp, on=['UID'], how='left')
    data['action_nonan_'+col+'_ratio'] = data['action_nonan_'+col+'_count']/(0.01+data['action_cnt'])

#### 用户行为地点发生在中国境内的频次及频率 ####
tmp = user_action[user_action['is_china']>=0].copy()
gp = tmp.groupby(['UID'])['is_china'].agg({'action_in_china_count':'sum',
                                           'action_in_china_ratio':'mean'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')

#### 用户当前行为是否为常用设备,统计频次及频率 ####
used_cols = ['device1', 'ip', 'ip_sub', 'mac1', 'device2','device_code','geo_code','province','city','district']
stats = user_action.copy()
for col in used_cols:
    gp = user_action.groupby(['UID', col]).size().reset_index().rename(columns={0: 'action_favor_'+col+'_count'})
    gp = gp.sort_values(by=['UID', 'action_favor_'+col+'_count'])
    gp = gp.groupby(['UID']).last().reset_index().rename(columns={col: 'action_favor_'+col})
    stats = stats.merge(gp, on=['UID'], how='left')
    stats['is_action_favor_'+col] = 0
    stats.loc[stats['action_favor_'+col]==stats[col], 'is_action_favor_'+col] = 1
    gp = stats.groupby(['UID'])['is_action_favor_'+col].agg({'is_action_favor_'+col+'_count': 'sum',
                                                             'is_action_favor_'+col+'_mean': 'mean'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')


####用户第一次操作/交易时间，最后一次操作/交易时间####
gp = user_action.groupby(['UID'])['actionTimestamp'].agg({'action_lastTimestamp': 'max',
                                                          'action_firstTimestamp': 'min'}).reset_index()
gp['action_timedelta'] = gp['action_lastTimestamp'] - gp['action_firstTimestamp']
data = data.merge(gp, on=['UID'], how='left')
gp = user_action[user_action['action_type']==2].groupby(['UID'])['actionTimestamp'].agg({'trade_lastTimestamp': 'max',
                                                                                         'trade_firstTimestamp': 'min'}).reset_index()
gp['trade_timedelta'] = gp['trade_lastTimestamp'] - gp['trade_firstTimestamp']
data = data.merge(gp, on=['UID'], how='left')

####用户每天操作/交易频次、转化率统计####
stats = user_action.groupby(['UID','day','action_type',]).size().unstack().fillna(0).reset_index()
stats.columns = ['UID', 'day', 'operation_day_count', 'trade_day_count']
stats['action_day_count'] = stats['operation_day_count']+stats['trade_day_count']
stats['trade_day_ratio'] = stats['trade_day_count']/stats['action_day_count']
gp = stats.groupby(['UID'])[['action_day_count', 'operation_day_count', 'trade_day_count', 'trade_day_ratio']]                                                                                            .agg({'max','min','std','mean','skew'})
gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')
data['action_day_count_diff'] = data['action_day_count_max'] - data['action_day_count_min']
data['trade_day_count_diff'] = data['trade_day_count_max'] - data['trade_day_count_min']
data['operation_day_count_diff'] = data['operation_day_count_max'] - data['operation_day_count_min']
data['trade_day_ratio_diff'] = data['trade_day_ratio_max'] - data['trade_day_ratio_min']

####用户操作到交易时间间隔小于100秒的次数####
user_action = user_action.sort_values(by=['actionTime'])
userid = user_action['UID'].unique()
timespancount_dict = {'UID': [],
                      'operation_to_trade_timdelta_count': []}
for uid in userid:
    action_df = user_action[user_action['UID']==uid].copy()
    actiontimespancount = futl.getActionTimeSpan(action_df, 1, 2, timethred = 100)
    timespancount_dict['UID'].append(uid)
    timespancount_dict['operation_to_trade_timdelta_count'].append(actiontimespancount)
timespancount_dict = pd.DataFrame(timespancount_dict)
data = data.merge(timespancount_dict, on=['UID'], how='left')

#### 设备、环境的热度(mean,max,min,skew,std,sum) ####
used_cols =['city', 'ip', 'nation','device2','geo_code', 'district','ip_sub','device1', 'mac1','device_brand','province','device_code']
for col in used_cols:
    stats = user_action.groupby([col])['UID'].nunique().reset_index().rename(columns={'UID': col+'_used_nunique'})
    tmp = user_action.merge(stats, on=[col], how='left')
    gp = tmp.groupby(['UID'])[col+'_used_nunique'].agg({col+'_used_nunique_max': 'max',
                                                        col+'_used_nunique_mean': 'mean',
                                                        col+'_used_nunique_min': 'min',
                                                        col+'_used_nunique_skew': 'skew',
                                                        col+'_used_nunique_mean': 'std'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')
    
####设备被用户操作的时间间隔####
used_cols = ['device_code', 'ip', 'ip_sub', 'mac1', 'device1','device2', 'geo_code']
for col in used_cols:
    data = futl.user_timedelta(data,user_action,col)
    
#### 用户的交易总金额 ####
gp = transaction_df.groupby('UID')['trans_amt'].agg({'trans_amt_mean':'mean',
                                                      'trans_amt_sum':'sum',
                                                      'trans_amt_max':'max',
                                                      'trans_amt_min':'min',
                                                      'trans_amt_std':'std',
                                                      'trans_amt_std':'skew'})
data = data.merge(gp,on = 'UID',how = 'left')
    
####每小时/半夜/早上/下午/晚上/1/3/7天的count/count_gap/count_rate####
for col in ['mac1', 'mac2', 'ip_sub']:
    data = futl.freq_time(data, operation_df, col, 'op')

for col in ['merchant', 'mac1', 'ip_sub', 'market_code']:
    data = futl.freq_time(data, transaction_df, col, 'tr')
    
####每小时/半夜/早上/下午/晚上/1/3/7天该特征进行交易的钱数的的count/count_gap/count_rate####
for col in ['merchant', 'mac1', 'ip_sub', 'market_code']:
    data = futl.transtime_freq(data, transaction_df, col)

#### 用户使用的资金来源、交易方式、交易商家、交易金额、账户、转出账户、转入账户、营销活动种类个数 ####
cols = ['amt_src1', 'amt_src2', 'trans_type1', 'trans_type2', 'merchant',              'trans_amt', 'acc_id1', 'acc_id2', 'acc_id3', 'market_code', 'market_type']
for col in cols:
    gp = transaction_df.groupby('UID')[col].nunique().reset_index().rename(columns={col:'trade_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')

#### 用户当前交易行为是否使用常用的资金来源、交易方式、交易商家、账户 ####
cols = ['amt_src1', 'trans_type1', 'merchant', 'acc_id1']
stats = transaction_df.copy()
for col in cols:
    gp = transaction_df.groupby(['UID', col]).size().reset_index().rename(columns={0: 'trade_favor_'+col+'_count'})
    gp = gp.sort_values(by=['UID', 'trade_favor_'+col+'_count'])
    gp = gp.groupby(['UID']).last().reset_index().rename(columns={col: 'trade_favor_'+col})
    stats = stats.merge(gp, on=['UID'], how='left')
    stats['is_trade_favor_'+col] = 0
    stats.loc[stats['trade_favor_'+col]==stats[col], 'is_trade_favor_'+col] = 1
    gp = stats.groupby(['UID'])['is_trade_favor_'+col].agg({'is_trade_favor_'+col+'_count': 'sum',
                                                            'is_trade_favor_'+col+'_mean': 'mean'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')

#### 用户在该auxicols中用该支付方式(maincols)支付次数,时间频率：每小时 ####
auxicols = ['ip_sub','mac1','device_code','ip']
maincols = ['trans_type1','market_code']
data = futl.freq_pay_time(data,transaction_df,auxicols,maincols)

#### 用户在用该支付方式(maincols)支付次数,时间频率：早中晚半夜每小时每1/3/7天 ####
maincols = ['trans_type1','market_code']
data = futl.freq_pay_time2(data, transaction_df, maincols)

####  用户在该auxicol 和strans_amt的时间频率统计 ####
maincols = ['','merchant','mac1', 'ip_sub', 'trans_type1', 'market_code']
for maincol in maincols:
    data = futl.transamt_timefreq(data, transaction_df, maincol,'trans_amt')
    
#### 用户时间频率 ####
data = futl.user_freqtime(data, operation_df)
data = futl.user_freqtime(data, transaction_df)

#### x1和x2的交叉count ####
opcols = ['ip', 'geo_code','ip_sub','wifi']
trcols = ['merchant', 'geo_code','ip_sub','code1']
for x1 in opcols:
    for x2 in opcols:
        if x1 != x2:
            data = futl.get_two_count(data,operation_df,x1,x2,'op')
            data = futl.get_two_shuxing_count(data,operation_df,x1,x2,'op')
for x1 in trcols:
    for x2 in trcols:
        if x1 != x2:
            data = futl.get_two_count(data,transaction_df,x1,x2,'tr')
            data = futl.get_two_shuxing_count(data,transaction_df,x1,x2,'tr')

data = futl.get_three_count(data, transaction_df, 'merchant', 'day', 'device1', 'tr')
data = futl.get_three_count(data, transaction_df, 'merchant', 'day', 'device2', 'tr')


data.to_csv('./data/data.csv')

