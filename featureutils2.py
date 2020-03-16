#!/usr/bin/env python
# coding: utf-8

import datetime
import numpy as np
import pandas as pd

def dayandtime2date(dayint,timestr):
    start_datestr = "2018-10-31"
    start_date = datetime.datetime.strptime(start_datestr, '%Y-%m-%d')
    timestrlist = [int(x) for x in timestr.split(":")]
    del start_datestr
    datadate = start_date + datetime.timedelta(days=dayint)
    return datetime.datetime.combine(datadate, datetime.time(*timestrlist))

def dif_to_sec(x):
    try:
        d = abs(x).seconds
    except Exception as e:
        d = np.nan
    return d

def diffWindow600Secd(train_oper,justTran):
    '''
    > datadate: combined day and time base on 2018/10/31 tran:1-30 test 31:61
    > time_group_cumcount: use cumcount generate user oper in window of 600 seconds time diff
    > useful : time_diff / time_group_cumcount
    '''
    train_oper["datadate"] = train_oper.apply(lambda x: dayandtime2date(x['day_new'],x['time']),axis=1)
    train_oper.sort_values(['UID', 'datadate'], inplace=True)
    train_oper["time_diff"] = train_oper.groupby("UID")["datadate"].diff(periods=-1) #上一条数据和下一条数据的时间差
    train_oper["time_diff_secd"] = train_oper["time_diff"].map(dif_to_sec) # 把时间差转换成秒
    train_oper["time_diff_secd"]= train_oper["time_diff_secd"].fillna(99999)
    if justTran:
        train_oper["trans_amt_dif"] = train_oper.groupby("UID")["trans_amt"].diff(periods=1) #后一个数减前一个数
        train_oper["trans_amt_dif"]= train_oper["trans_amt_dif"].fillna(0)
        train_oper["bal_dif"] = train_oper.groupby("UID")["bal"].diff(periods=1)
        train_oper["bal_dif"]= train_oper["bal_dif"].fillna(0)
        
    train_oper["time_group_tmp"] = train_oper["time_diff_secd"].map(lambda x: 1 if x>600 else np.nan)
    train_oper["time_group_tmp_cumcount"] = train_oper.groupby(["UID","time_group_tmp"]).cumcount()
    train_oper["time_group_tmp_cumcount"] = train_oper["time_group_tmp_cumcount"] *  train_oper["time_group_tmp"] 
    train_oper["time_group"] = train_oper["time_group_tmp_cumcount"].fillna(method='bfill')  #backfill/bfill：用下一个非缺失值填充该缺失值
    train_oper = train_oper.drop(["time_group_tmp" , "time_group_tmp_cumcount","time_diff"],axis=1)
    return train_oper

def get_hour(x):
    try:
        h = x.hour
    except Exception as e:
        h = np.nan
    return h