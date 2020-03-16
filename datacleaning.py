#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
sns.set(style = 'white', context = 'notebook', palette = 'deep')
sns.set_style('white')

# 固定随机性
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=0)

### 读取数据
data = pd.read_csv('./data/data.csv')
train = data[data['Tag']!=0.5].copy()
test = data[data['Tag']==0.5].copy()
tag = train['Tag']
del train['Unnamed: 0'],train['Unnamed: 0.1']
del test['Unnamed: 0'],test['Unnamed: 0.1']
feature = train.columns.difference(['UID', 'Tag']).tolist()
train.set_index(['UID'],inplace = True)

###删除缺失值大于95%的特征
missingrate = pd.DataFrame(train.isna().sum() / len(train),columns = ['ratio']).sort_values(by = 'ratio',ascending=False)# missing rate
delete_cols = missingrate[missingrate['ratio']>0.95].index.tolist() # 缺失值大于95%的column
used_cols = [i for i in feature if i not in delete_cols]

### 按行统计缺失值
plt.figure(figsize=[14,5])

plt.subplot(1,2,1)
train_row = pd.DataFrame(train[feature].isna().sum(axis=1),columns=['missnum']).sort_values(by='missnum',ascending=True).reset_index()
del train_row['index']
plt.scatter(train_row.index,train_row['missnum'])
plt.ylim(0,2100) 
plt.title('train set')
plt.xlabel('Order Number(sort increasingly)')
plt.ylabel('Number of Missing Attributes') 

plt.subplot(1,2,2)
test_row = pd.DataFrame(test[feature].isna().sum(axis=1),columns=['missnum']).sort_values(by='missnum',ascending=True).reset_index()
del test_row['index']
plt.scatter(test_row.index,test_row['missnum'])
plt.ylim(0,2100) 
plt.title('test set')
plt.xlabel('Order Number(sort increasingly)')
plt.ylabel('Number of Missing Attributes');



# 删除异常样本
train.set_index(['UID'],inplace = True)
train_row = pd.DataFrame(train[feature].isna().sum(axis=1),columns=['missnum']).reset_index
train = train.reset_index()
train = train.merge(train_row,on='UID',how='left')
train = train[train['missnum']>2000]
del train['missnum']

# 剔除常变量，删除特征std接近0的特征
feature_std = pd.DataFrame(train[used_cols].describe())
feature_std = pd.DataFrame(feature_std.loc['std',:],columns=['std']).sort_values(by='std',ascending=False) #std均较大


def XGB_test(train_x,train_y,test_x,test_y):  
    from multiprocessing import cpu_count  
    clf = xgb.XGBClassifier(boosting_type='gbdt', num_leaves=31,
                            reg_alpha=0.0, reg_lambda=1,
                            max_depth=2, n_estimators=800,
                            max_features = 140, objective='binary:logistic',
                            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.05, min_child_weight=50,
                            n_jobs=cpu_count()-1,
                            num_iterations = 800 #迭代次数  
    )  
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)  
    return clf

train_x,val_x,train_y,val_y = train_test_split(train[used_cols],tag,test_size=0.2)
XGBmodel = XGB_test(train_x,train_y,val_x,val_y)
col_importance = pd.DataFrame(
                    {'name':XGBmodel.get_booster().feature_names,
                     'importance':XGBmodel.feature_importances_
    }).sort_values(by=['importance'],ascending=False)

choose_cols = col_importance[col_importance['importance']>0]['name'].tolist()#筛选特征

# 去掉离群值
train.set_index(['UID'],inplace = True)
check_cols = col_importance.head(30)['name'].tolist()
check_df = pd.DataFrame(train[check_cols].isna().sum(axis=1),columns=['missnum']).reset_index()
train = train.reset_index()
train = train.merge(check_df,on='UID',how='left')
train = train[train['missnum']<=15]
del train['missnum']

path = './data'
with open(path+'/xgbfeatures.txt','w') as f:
    f.write(str(choose_cols))

train.to_csv('./data/train_after_cleaning.csv',index=False)
test.to_csv('./data/test_after_cleaning.csv',index=False)

