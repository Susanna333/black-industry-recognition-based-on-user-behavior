#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import datetime

def group_diff_time(data, key, value, n, name):
    '''
    计算分组时间差
    '''
    data_temp = data[key + [value]].copy()
    shift_value = data_temp.groupby(key)[value].shift(n)
    data_temp[name] = data_temp[value] - shift_value
    return data_temp[name]

def user_timedelta(data,user_action,col):
    '''
   计算设备被用户操作的时间间隔 
    '''
    stats = user_action[user_action[col].isnull()==False][['UID', 'actionTimestamp']+[col]].copy()
    stats = stats.sort_values(by=[col, 'actionTimestamp'])
    stats['timedelta'] = group_diff_time(stats, [col], 'actionTimestamp', 1, 'timedelta')
    gp = stats.groupby(col)['actionTimestamp'].agg({col+'_used_user_actionTimestamp_min':'min',
                                                    col+'_used_user_actionTimestamp_max':'max'}).reset_index()
    gp[col+'_used_user_actionTimestamp_timedelta'] = gp[col+'_used_user_actionTimestamp_max'] - gp[col+'_used_user_actionTimestamp_min']
    tmp = user_action.merge(gp, on=[col], how='left')
    gp = tmp.groupby(['UID'])[col+'_used_user_actionTimestamp_timedelta'].agg({col+'_used_user_actionTimestamp_timedelta_max': 'max',                                                                   col+'_used_user_actionTimestamp_timedelta_min': 'min',                                                                   col+'_used_user_actionTimestamp_timedelta_mean': 'mean',                                                                 col+'_used_user_actionTimestamp_timedelta_skew': 'skew',
                                                      col+'_used_user_actionTimestamp_timedelta_std': 'std'})
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')

    
    
    gp = stats.groupby([col])['timedelta'].agg({col+'_used_timedelta_mean': 'mean',
                                                col+'_used_timedelta_skew': 'skew',
                                                col+'_used_timedelta_std': 'std',
                                                col+'_used_timedelta_max': 'max',
                                                col+'_used_timedelta_min': 'min',}).reset_index()
    tmp = user_action.merge(gp, on=[col], how='left')
    columns = [f for f in gp.columns if f not in ['UID']]
    gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'mean', 'skew', 'std'})
    gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')
    return data


def freq_time(data, temp, col, t):
    """
    label和时间交叉count
    """
    # 每一天的每一小时的count
    gp = temp[[col, 'day', 'hour']]
    gp[str(col) + '_hour_count'] = 1
    gp = gp.groupby([col, 'day', 'hour']).agg('count').reset_index()
    gp1 = gp.copy()
    gp1 = gp1.merge(temp[['UID', col, 'day', 'hour']], on=[col, 'day', 'hour'], how='left')
    gp1 = gp1[['UID', str(col) + '_hour_count']]
    gp1 = gp1.groupby('UID')[str(col) + '_hour_count'].agg({'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp1.columns if i not in ['UID']]
    b = [t+str(col)+'_hour_'+str(i) for i in gp1.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp1 = gp1.rename(columns = listdict)
    data = data.merge(gp1,on='UID',how='left')
    
    # 每一天的每一小时的count_gap
    gp['before_hour_count'] = gp.groupby(col)[str(col) + '_hour_count'].shift(1)
    gp2 = gp.copy()
    gp2[col + 'hour_count_gap'] = gp2[str(col) + '_hour_count'] - gp2['before_hour_count']
    gp2 = gp2.drop([str(col) + '_hour_count', 'before_hour_count'], axis=1)
    gp2 = gp2.merge(temp[['UID', col, 'day', 'hour']], on=[col, 'day', 'hour'], how='left')
    gp2 = gp2[['UID', str(col) + 'hour_count_gap']]
    gp2 = gp2.groupby('UID')[str(col) + 'hour_count_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp2.columns if i not in ['UID']]
    b = [t+str(col)+'_hour_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp2 = gp2.rename(columns = listdict)
    data = data.merge(gp2,on='UID',how='left')
    
    # 每一天的每一小时的count_rate
    gp[str(col) + 'hour_count_rate'] = gp[str(col) + '_hour_count'] / gp['before_hour_count']
    gp = gp.drop([str(col) + '_hour_count', 'before_hour_count'], axis=1)
    gp = gp.merge(temp[['UID', col, 'day', 'hour']], on=[col, 'day', 'hour'], how='left')
    gp = gp[['UID', str(col) + 'hour_count_rate']]
    gp = gp.groupby('UID')[str(col) + 'hour_count_rate'].agg('sum').reset_index()
    a = [i for i in gp.columns if i not in ['UID']]
    b = [t+str(col)+'hour_count_rate']
    listdict = dict(zip(a,b))
    gp = gp.rename(columns = listdict)
    data = data.merge(gp,on='UID',how='left')
    
    
    #半夜：hour <= 5
    #早上：hour >= 6 & hour <= 11
    #下午：hour >= 12 & hour <= 17
    #晚上：hour >= 18 & hour <= 23
    morning = temp[temp.hour <= 5]
    work_time = temp[(temp.hour >= 6)&(temp.hour <= 11)]
    afternoon = temp[(temp.hour >= 12)&(temp.hour <= 17)]
    night = temp[(temp.hour >= 18)&(temp.hour <= 23)]
    status = {'morning':morning,
              'work_time':work_time,
              'afternoon':afternoon,
              'night':night,
              'day':temp
             }
    for name,statu in status.items():
        # 半夜/早上/下午/晚上/一天的count
        gp = statu[[col, 'day']]
        gp[str(col) + '_count'] = 1
        gp = gp.groupby([col, 'day']).agg('count').reset_index()
        gp = gp.merge(temp[['UID', col, 'day']], on=[col, 'day'], how='left')
        gp = gp[['UID', str(col) + '_count']]
        gp = gp.groupby('UID')[str(col) + '_count'].agg({'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp.columns if i not in ['UID']]
        b = [t+str(col)+'_'+ name +'_'+str(i) for i in gp.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp = gp.rename(columns = listdict)
        data = data.merge(gp, on='UID', how='left')
        
        # 半夜/早上/下午/晚上/一天的gap
        gp = statu[[col, 'day']]
        gp[str(col) + '_gap_count'] = 1
        gp = gp.groupby([col,'day']).agg('count').reset_index()
        gp = gp.merge(temp[['UID', col, 'day']], on=[col, 'day'], how='left')
        gp[str(col) + '_gap_count_before'] = gp.groupby('UID')[str(col) + '_gap_count'].shift(1)
        gp1 = gp.copy()
        gp1[str(col) + '_gap_count_gap'] =gp1[str(col) + '_gap_count'] - gp1[str(col) + '_gap_count_before']
        gp1 = gp1[['UID', str(col) + '_gap_count_gap']]
        gp1 = gp1.groupby('UID')[str(col) + '_gap_count_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        b = [t+str(col)+'_'+ name +'_gap_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')
        
        # 半夜/早上/下午/晚上/一天的rate
        gp = statu[[col, 'day']]
        gp[str(col) + '_gap_count'] = 1
        gp = gp.groupby([col,'day']).agg('count').reset_index()
        gp = gp.merge(temp[['UID', col, 'day']], on=[col, 'day'], how='left')
        gp[str(col) + '_gap_count_before'] = gp.groupby('UID')[str(col) + '_gap_count'].shift(1)
        gp[str(col) + '_rate_count_rate'] =gp[str(col) + '_gap_count'] / gp[str(col) + '_gap_count_before']
        gp = gp[['UID',str(col) + '_rate_count_rate']]
        gp = gp.groupby('UID')[str(col) + '_rate_count_rate'].agg('sum').reset_index()
        a = [i for i in gp.columns if i not in ['UID']]
        b = [t+str(col)+'_'+name + 'rate_count_rate']
        listdict = dict(zip(a,b))
        gp = gp.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')
      
    
    windowtime=[3,7]
    for tw in windowtime:
        # 每3/7天的count_gap
        gp = temp[[col, 'day']]
        gp[str(col) + '_day_gap_count'] = 1
        gp = gp.groupby([col, 'day']).agg('count').reset_index()
        gp['before_day_count'] = gp.groupby(col)[str(col) + '_day_gap_count'].shift(tw)
        gp[str(col) + 'day_count_gap'] = gp[str(col) + '_day_gap_count'] - gp['before_day_count']
        gp = gp.drop([str(col) + '_day_gap_count', 'before_day_count'], axis=1)
        gp = gp.merge(temp[['UID', col, 'day']], on=[col, 'day'], how='left')
        gp = gp[['UID', str(col) + 'day_count_gap']]
        gp = gp.groupby('UID')[str(col) + 'day_count_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp.columns if i not in ['UID']]
        b = [t+str(col)+'_'+str(tw)+'day_gap_'+str(i) for i in gp.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp = gp.rename(columns = listdict)
        data = data.merge(gp, on='UID', how='left')

        # 每3/7天的count_rate
        gp = temp[[col, 'day']]
        gp[str(col) + '_day_rate_count'] = 1
        gp = gp.groupby([col, 'day']).agg('count').reset_index()
        gp['before_day_count'] = gp.groupby(col)[str(col) + '_day_rate_count'].shift(tw)
        gp[str(col)+ 'day_count_rate'] = gp[str(col) + '_day_rate_count'] / gp['before_day_count']
        gp = gp.drop([str(col) + '_day_rate_count', 'before_day_count'], axis=1)
        gp = gp.merge(temp[['UID', col, 'day']], on=[col, 'day'], how='left')
        gp = gp[['UID', str(col) + 'day_count_rate']]
        gp = gp.groupby('UID')[str(col) + 'day_count_rate'].agg('sum').reset_index()
        a = [i for i in gp.columns if i not in ['UID']]
        b = [t+str(col)+'_'+str(tw) + 'day_count_rate']
        listdict = dict(zip(a,b))
        gp = gp.rename(columns = listdict)
        data = data.merge(gp, on='UID', how='left')
    
    print(col + ' of ' + t, 'is over!')
    return data



def transtime_freq(data, temp, x):
    # 每天的每一小时用该x(如ip)进行交易的钱数
    gp = temp[[x, 'day', 'hour', 'trans_amt']]
    gp['every_day_' + x + '_amt'] = gp['trans_amt']
    gp = gp.groupby([x, 'day', 'hour'])['every_day_' + x + '_amt'].agg('sum').reset_index()
    gp = gp.merge(temp[['UID', x, 'day', 'hour']], on=[x, 'day', 'hour'], how='left')
    gp = gp[['UID', 'every_day_' + x + '_amt']]
    gp1 = gp.copy()
    gp1 = gp1.groupby('UID')['every_day_' + x + '_amt'].agg({'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp1.columns if i not in ['UID']]
    b = [str(x)+'_hour_amt_'+str(i) for i in gp1.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp1 = gp1.rename(columns = listdict)
    data = data.merge(gp1, on='UID', how='left')
    
    # 每天的每一小时用该x(如ip)进行交易的钱数的gap
    gp['before_amt'] = gp.groupby('UID')['every_day_' + x + '_amt'].shift(1)
    gp2 = gp.copy()
    gp2[x + '_hour_amt_gap_'] = gp2['every_day_' + x + '_amt'] - gp2['before_amt']
    gp2 = gp2[['UID', x + '_hour_amt_gap_']]
    gp2 = gp2.groupby('UID')[x + '_hour_amt_gap_'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp2.columns if i not in ['UID']]
    b = [str(x)+'_hour_amt_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp2 = gp2.rename(columns = listdict)
    data = data.merge(gp2, on='UID', how='left')
    
     # 每天的每一小时用该x(如ip)进行交易的钱数的rate
    gp[x + '_hour_amt_rate'] = gp['every_day_' + x + '_amt'] / gp['before_amt']
    gp = gp[['UID', x + '_hour_amt_rate']]
    gp = gp.groupby('UID')[x + '_hour_amt_rate'].agg('sum').reset_index()
    data = data.merge(gp, on='UID', how='left')
    
    #半夜：hour <= 5
    #早上：hour >= 6 & hour <= 11
    #下午：hour >= 12 & hour <= 17
    #晚上：hour >= 18 & hour <= 23
    morning = temp[temp.hour <= 5]
    work_time = temp[(temp.hour >= 6)&(temp.hour <= 11)]
    afternoon = temp[(temp.hour >= 12)&(temp.hour <= 17)]
    night = temp[(temp.hour >= 18)&(temp.hour <= 23)]
    status = {'morning':morning,
              'work_time':work_time,
              'afternoon':afternoon,
              'night':night,
              'day':temp
             }
    for name,statu in status.items():
        # 半夜/早上/下午/晚上/一天
        gp = statu[[x, 'day', 'trans_amt']]
        gp[x + '_amt'] = gp['trans_amt']
        gp = gp.groupby([x, 'day'])[x + '_amt'].agg('sum').reset_index()
        gp = gp.merge(temp[['UID', x, 'day']], on=[x, 'day'], how='left')
        gp = gp[['UID', x + '_amt']]
        gp1 = gp.copy()
        gp1 = gp1.groupby('UID')[x + '_amt'].agg({'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        b = [str(x)+'_'+ name +'_amt_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp1 = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')
        # gap
        gp['before_amt'] = gp.groupby('UID')[x + '_amt'].shift(1)
        gp2 = gp.copy()
        gp2[x + '_amt_gap'] = gp2[x + '_amt'] - gp2['before_amt']
        gp2 = gp2[['UID', x + '_amt_gap']]
        gp2 = gp2.groupby('UID')[x + '_amt_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp2.columns if i not in ['UID']]
        b = [str(x)+'_'+name+'_amt_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp2 = gp2.rename(columns = listdict)
        data = data.merge(gp2, on='UID', how='left')
        # rate
        gp[x + '_' + name+ '_amt_rate'] = gp[x + '_amt']/gp['before_amt']
        gp = gp[['UID', x + '_' + name+ '_amt_rate']]
        gp = gp.groupby('UID')[x + '_' + name+ '_amt_rate'].agg('sum').reset_index()
        data = data.merge(gp, on='UID', how='left')
        
        
    windowtime=[3,7]
    for tw in windowtime:
        # 每3/7天的gap
        gp = temp[[x, 'day', 'trans_amt']]
        gp[x + '_amt'] = gp['trans_amt']
        gp = gp.groupby([x, 'day'])[x + '_amt'].agg('sum').reset_index()
        gp = gp.merge(temp[['UID', x, 'day']], on=[x, 'day'], how='left')
        gp = gp[['UID', x + '_amt']]
        gp['before_amt'] = gp.groupby('UID')[x + '_amt'].shift(tw)
        gp1 = gp.copy()
        gp1[x + '_amt_gap'] = gp1[x + '_amt'] - gp1['before_amt']
        gp1 = gp1[['UID',x + '_amt_gap']]
        gp1 = gp1.groupby('UID')[x + '_amt_gap'].agg({'sum','mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        b = [str(x)+'_'+ str(tw) +'day_amt_gap_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp1 = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')
        
        # gap
        gp[x + '_'+ str(tw)+'day_amt_rate_'] = gp[x + '_amt'] / gp['before_amt']
        gp = gp[['UID',x + '_'+ str(tw)+'day_amt_rate_']]
        gp = gp.groupby('UID')[x + '_'+ str(tw)+'day_amt_rate_'].agg({'sum'}).reset_index()
        data = data.merge(gp, on='UID', how='left')
        
    print(x, 'is over!')
    return data



def getActionTimeSpan(df, actiontypeA, actiontypeB, timethred):
    '''
    计算行为A到行为B时间间隔小于阈值的个数
    '''
    timespan_list = []
    i = 0
    while i < (len(df) - 1):
        if df['action_type'].iat[i] == actiontypeA:
            timeA = df['actionTimestamp'].iat[i]
            for j in range(i + 1, len(df)):
                if df['action_type'].iat[j] == actiontypeA:
                    timeA = df['actionTimestamp'].iat[j]
                else: 
                    timeB = df['actionTimestamp'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i += 1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)


def calculate_intersection_ratio(x, y):
    '''
    计算两个序列之间重合度
    '''
    try:
        inter = x.intersection(y)
        union = x.union(y)
        return 1.0*len(inter)/len(union)
    except:
        return np.nan 


def freq_pay_time(data, temp, auxicols, maincols):
    timelist = {'hour':['day','hour'],
                    'day':['day']}
    # 用户在该ip用该支付方式支付次数
    for maincol in maincols:
        for auxicol in auxicols:   
            for i,time in timelist.items():
                col_list =time + ['UID',auxicol, maincol]
                gp = temp[col_list]
                gp['count'] = 1
                gp = gp.groupby(col_list).agg('count').reset_index()
                gp = gp[['UID','count']]
                gp = gp.groupby('UID')['count'].agg({'mean', 'max', 'min', 'std'}).reset_index()
                a = [i for i in gp.columns if i not in ['UID']]
                b = [maincol+'_'+auxicol+'_'+i+'_count_'+str(i) for i in gp.columns if i not in ['UID']]
                listdict = dict(zip(a,b))
                gp = gp.rename(columns = listdict)
                data = data.merge(gp,on='UID',how='left')
    return data

def freqshift(data,statu,time,name,tw,*maincol):
    col_list =time + ['UID', maincol]
    # count
    gp = statu[col_list]
    gp['count'] = 1
    gp = gp.groupby(col_list).agg('count').reset_index()
    if tw==1:
        gp = gp[['UID', 'count']]
        gp1 = gp.copy()
        gp1 = gp1.groupby('UID')['count'].agg({'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        b = [name+'_'+maincol+'_count_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp1 = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')

    # gap
    gp['gap_count_before'] = gp.groupby('UID')['count'].shift(tw)
    gp2 = gp.copy()
    gp2['gap_count_gap'] =gp2['count'] - gp2['gap_count_before']
    gp2 = gp2[['UID', 'gap_count_gap']]
    gp2 = gp2.groupby('UID')['gap_count_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp2.columns if i not in ['UID']]
    if tw == 1:
        b = [name+'_'+maincol+'_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    else:
        b = [maincol+'_'+str(tw)+'_day_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp2 = gp2.rename(columns = listdict)
    data = data.merge(gp2, on='UID', how='left')

    # rate
    gp[maincol+'_rate'] =gp['count'] / gp['gap_count_before']
    gp = gp[['UID',maincol+'_rate']].groupby('UID')[maincol+'_rate'].agg('sum').reset_index()
    if tw==1:
        gp = gp.rename(columns={maincol+'_rate':name+'_'+maincol+'_rate'})
    else:
        gp = gp.rename(columns={maincol+'_rate':maincol+'_'+str(tw)+'_day_rate'})
    data = data.merge(gp, on='UID', how='left')
    return data

def freq_pay_time2(data, temp, *maincols):
    timelist = {'hour':['day','hour'],
                'day':['day']}
    morning = temp[temp.hour <= 5]#半夜：hour <= 5
    work_time = temp[(temp.hour >= 6)&(temp.hour <= 11)]#早上：hour >= 6 & hour <= 11
    afternoon = temp[(temp.hour >= 12)&(temp.hour <= 17)]#下午：hour >= 12 & hour <= 17
    night = temp[(temp.hour >= 18)&(temp.hour <= 23)]#晚上：hour >= 18 & hour <= 23
    status = {'morning':morning,
              'work_time':work_time,
              'afternoon':afternoon,
              'night':night,
              'hour':temp
             }
    windowtime=[3,7]
    
    for maincol in maincols:
        for name,statu in status.items():
            tw = 1
            if name == 'hour':
                for key,time in timelist.items():
                    data = freqshift(data,statu,time,key,tw,maincol)
            else:
                time = ['day','hour']
                data = freqshift(data,statu,time,name,tw,maincol)
        
        for tw in windowtime:
            time = ['day']
            data = freqshift(maincol,data,temp,time,'',tw)
        print(maincol, 'is over!')
    
    return data



def transamtshift(data,statu,time,maincol,auxicol,name,tw):
    col_list =time + ['UID', maincol]
    col_list.drop('')
    # count
    gp = statu[col_list+[auxicol]]
    gp = gp.groupby(col_list)[auxicol].agg('sum').reset_index()
    gp = gp[['UID', auxicol]]
    if tw==1:
        gp1 = gp.copy()
        gp1 = gp1.groupby('UID')[auxicol].agg({'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        if maincol == '':
            b = ['money_frequence_'+name+'_'+str(i) for i in gp1.columns if i not in ['UID']]
        else:
            b = [maincol+'_'+name+'_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp1 = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')

    # gap
    gp['before'] = gp.groupby('UID')[auxicol].shift(tw)
    gp2 = gp.copy()
    gp2['gap'] =gp2[auxicol] - gp2['before']
    gp2 = gp2[['UID', 'gap']]
    gp2 = gp2.groupby('UID')['gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp2.columns if i not in ['UID']]
    if tw == 1:
        if maincol == '':
            b = ['money_frequence_'+name+'_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
        else:
            b = [maincol+'_'+name+'_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    else:
        if miancol == '':
            b = ['money_frequence_'+str(tw)+'_day_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
        else:
            b = [maincol+'_'+str(tw)+'_day_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp2 = gp2.rename(columns = listdict)
    data = data.merge(gp2, on='UID', how='left')

    # rate
    gp['rate'] =gp[auxicol] / gp['before']
    gp = gp[['UID','rate']].groupby('UID')['rate'].agg('sum').reset_index()
    if tw==1:
        if maincol == '':
            gp = gp.rename(columns={'rate':'money_frequence_'+name+'_rate'})
        else:
            gp = gp.rename(columns={'rate':maincol+'_'+name+'_rate'})
    else:
        if maincol == '':
            gp = gp.rename(columns={'rate':'money_frequence_'+str(tw)+'_day_rate'})
        else:
            gp = gp.rename(columns={'rate':maincol+'_'+str(tw)+'_day_rate'})
    data = data.merge(gp, on='UID', how='left')
    return data



def transamt_timefreq(data, temp, maincol,auxicol):
    """
    用户 * merchant时间频率
    """
    timelist = {'hour':['day','hour'],
                'day':['day']}
    morning = temp[temp.hour <= 5]#半夜：hour <= 5
    work_time = temp[(temp.hour >= 6)&(temp.hour <= 11)]#早上：hour >= 6 & hour <= 11
    afternoon = temp[(temp.hour >= 12)&(temp.hour <= 17)]#下午：hour >= 12 & hour <= 17
    night = temp[(temp.hour >= 18)&(temp.hour <= 23)]#晚上：hour >= 18 & hour <= 23
    status = {'morning':morning,
           'work_time':work_time,
           'afternoon':afternoon,
           'night':night,
           'hour':temp
             }
    windowtime=[3,7]
    
    for name,statu in status.items():
        tw = 1
        if name == 'hour':
            for key,time in timelist.items():
                data = transamtshift(data,statu,time,maincol,auxicol,key,tw) #每小时/每天
        else:
            time = ['day','hour']
            data = transamtshift(data,statu,time,maincol,auxicol,name,tw) #早中晚半夜

    for tw in windowtime:
        time = ['day']
        data = transamtshift(data,statu,time,maincol,auxicol,'',tw) #每3/7天
    print(maincol, 'is over!')
    
    return data

def freqshift2(data,statu,time,name,tw):
    col_list =time + ['UID']
    # count
    gp = statu[col_list]
    gp['count'] = 1
    gp = gp.groupby(col_list).agg('count').reset_index()
    if tw==1:
        gp = gp[['UID', 'count']]
        gp1 = gp.copy()
        gp1 = gp1.groupby('UID')['count'].agg({'mean', 'max', 'min', 'std'}).reset_index()
        a = [i for i in gp1.columns if i not in ['UID']]
        b = ['frequence_'+name+'_'+str(i) for i in gp1.columns if i not in ['UID']]
        listdict = dict(zip(a,b))
        gp1 = gp1.rename(columns = listdict)
        data = data.merge(gp1, on='UID', how='left')

    # gap
    gp['gap_count_before'] = gp.groupby('UID')['count'].shift(tw)
    gp2 = gp.copy()
    gp2['gap_count_gap'] =gp2['count'] - gp2['gap_count_before']
    gp2 = gp2[['UID', 'gap_count_gap']]
    gp2 = gp2.groupby('UID')['gap_count_gap'].agg({'sum', 'mean', 'max', 'min', 'std'}).reset_index()
    a = [i for i in gp2.columns if i not in ['UID']]
    if tw == 1:
        b = ['frequence_'+name+'_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    else:
        b = ['frequence_'+str(tw)+'_day_gap_'+str(i) for i in gp2.columns if i not in ['UID']]
    listdict = dict(zip(a,b))
    gp2 = gp2.rename(columns = listdict)
    data = data.merge(gp2, on='UID', how='left')

    # rate
    gp['_rate'] =gp['count'] / gp['gap_count_before']
    gp = gp[['UID','_rate']].groupby('UID')['_rate'].agg('sum').reset_index()
    if tw==1:
        gp = gp.rename(columns={'_rate':'frequence_'+name+'_rate'})
    else:
        gp = gp.rename(columns={maincol+'_rate':'frequence_'+str(tw)+'_day_rate'})
    data = data.merge(gp, on='UID', how='left')
    return data

def user_freqtime(data, temp):
    timelist = {'hour':['day','hour'],
                'day':['day']}
    morning = temp[temp.hour <= 5]#半夜：hour <= 5
    work_time = temp[(temp.hour >= 6)&(temp.hour <= 11)]#早上：hour >= 6 & hour <= 11
    afternoon = temp[(temp.hour >= 12)&(temp.hour <= 17)]#下午：hour >= 12 & hour <= 17
    night = temp[(temp.hour >= 18)&(temp.hour <= 23)]#晚上：hour >= 18 & hour <= 23
    status = {'morning':morning,
              'work_time':work_time,
              'afternoon':afternoon,
              'night':night,
              'hour':temp
             }
    windowtime=[3,7]
      
    for name,statu in status.items():
        tw = 1
        if name == 'hour':
            for key,time in timelist.items():
                data = freqshift2(data,statu,time,key,tw)
        else:
            time = ['day','hour']
            data = freqshift2(data,statu,time,name,tw)

    for tw in windowtime:
        time = ['day']
        data = freqshift2(data,temp,time,'',tw)
   
    return data


def get_two_count(data, t_data, x1, x2, t):
    """
    获得x1和x2的交叉count
    """
    temp = t_data[[x1, x2, 'UID']]
    
    operation_or_transaction_data = temp.copy()
    operation_or_transaction_data.drop_duplicates(inplace=True)
    
    temp[x1 + '_' + x2 + '_count'] = temp['UID']
    temp = temp.groupby([x1, x2])[x1 + '_' + x2 + '_count'].agg('nunique').reset_index()
    
    operation_or_transaction_data = pd.merge(operation_or_transaction_data, temp, on=[x1, x2], how='left')
    operation_or_transaction_data[x1 + '_' + x2 + '_count_uid_' + t] = 1
    operation_or_transaction_data = operation_or_transaction_data.groupby('UID')[x1 + '_' + x2 + '_count_uid_' + t].agg('sum').reset_index()
    
    data = data.merge(operation_or_transaction_data, on='UID', how='left')
    return data


def get_two_shuxing_count(data, t_data, x1, x2, t):
    """
    groupby(x1)[x2].agg('nunique').reset_index()
    """
    UID = t_data[['UID', x1]].drop_duplicates()
    temp = t_data[[x1, x2]]
    temp[x2 + '_of_nunique_' + x1] = temp[x2]
    temp = temp.groupby(x1)[x2 + '_of_nunique_' + x1].agg('nunique').reset_index()
    
    UID = pd.merge(UID, temp, on=x1, how='left')
    UID = UID.drop(x1, axis=1)
    UID = UID.groupby('UID')[x2 + '_of_nunique_' + x1].agg('sum').reset_index()
    
    data = pd.merge(data, UID, on='UID', how='left')
    return data

def get_three_count(data, t_data, x1, x2, x3, t):
    temp = t_data[[x1, x2, x3, 'UID']]
    
    operation_or_transaction_data = temp.copy()
    operation_or_transaction_data.drop_duplicates(inplace=True)
    
    temp[x1 + '_' + x2 + '_' + x3 + '_count'] = temp['UID']
    temp = temp.groupby([x1, x2, x3])[x1 + '_' + x2 + '_' + x3 + '_count'].agg('nunique').reset_index()
    
    operation_or_transaction_data = pd.merge(operation_or_transaction_data, temp, on=[x1, x2, x3], how='left')
    operation_or_transaction_data[x1 + '_' + x2 + '_' + x3 + '_count'] = 1
    operation_or_transaction_data = operation_or_transaction_data.groupby('UID')[x1 + '_' + x2 + '_' + x3 + '_count'].agg('sum').reset_index()
    
    print(operation_or_transaction_data)
    data = pd.merge(data, operation_or_transaction_data, on='UID', how='left')
    return data