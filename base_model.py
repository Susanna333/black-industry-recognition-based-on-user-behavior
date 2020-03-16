#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import os
import random
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from multiprocessing import cpu_count
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.metrics import make_scorer
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

# 读取数据
train = pd.read_csv('./data/train_after_cleaning.csv')
test = pd.read_csv('./data/test_after_cleaning.csv')
tag = train['Tag']
train_x = train[features]
test_x = test[features]

with open(r'./data/xgbfeatures.txt','r') as f:
    features = f.read()
features = eval(features)



### 贝叶斯优化
def LGB_bayesian(
    #learning_rate,
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    learning_rate,
    min_child_weight, 
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
   

    param = { 'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              'learning_rate' : learning_rate,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'save_binary': True,
              'seed': 1337,
              'feature_fraction_seed': 1337,
              'bagging_seed': 1337,
              'drop_seed': 1337,
              'data_random_seed': 1337,
              'boosting_type': 'gbdt',
              'verbose': 1,
              'is_unbalance': False,
              'boost_from_average': True,
              'metric':'auc'}    
    
    oof = np.zeros(len(train_x))
    trn_data= lgb.Dataset(train_x.iloc[bayesian_tr_idx][features].values, label=tag[bayesian_tr_idx])
    val_data= lgb.Dataset(train_x.iloc[bayesian_val_idx][features].values, label=tag[bayesian_val_idx])

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)
    
    oof[bayesian_val_idx]  = clf.predict(train_x.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(tag[bayesian_val_idx], oof[bayesian_val_idx])

    return score


bounds_LGB = {
    'num_leaves': (31, 500), 
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'learning_rate': (0.01, 0.3),
    'min_child_weight': (0.00001, 0.01),   
    'reg_alpha': (1, 8), 
    'reg_lambda': (0.5, 2),
    'max_depth':(-1,50)
}

bayesian_tr_idx, bayesian_val_idx = list(StratifiedKFold(n_splits=5,shuffle=True, random_state=0).split(train_x, tag))[0]

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=0)

LGB_BO.maximize()


# 评分算法
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

boparams = LGB_BO.max['params']
boparams['num_leaves'] = int(boparams['num_leaves'])
boparams['max_depth'] = int(boparams['max_depth'])
boparams['min_data_in_leaf'] = int(boparams['min_data_in_leaf'])
params2 = {'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
          'subsample_freq': 1,
          'seed': 0,
          'verbose': -1,
          'n_jobs': cpu_count()-1,
          }
boparams.update(params2)
parameters = boparams
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
xx_submit = []
xx_tpr = []
xx_auc = []
xx_iteration = []
oof_preds = np.zeros(train.shape[0])
feature_importance_df = pd.DataFrame()

for n_fold, (train_index, val_index) in enumerate(folds.split(train_x, tag)):
    dtrain = lgb.Dataset(data=train_x.iloc[train_index],
                         label=tag[train_index])
    dvalid = lgb.Dataset(data=train_x.iloc[val_index],
                         label=tag[val_index])
    clf = lgb.train(
        params=parameters,
        train_set=dtrain,
        num_boost_round=800,
        valid_sets=[dvalid],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)
    valid_preds = clf.predict(train_x.iloc[val_index], num_iteration=clf.best_iteration)
    print('Fold%2d LOGLOSS: %.6f' % (n_fold + 1, clf.best_score['valid_0']['binary_logloss']),           'Fold%2d TPR: %.6f' % (n_fold + 1, tpr_weight_funtion(tag[val_index], valid_preds)))
    xx_auc.append(clf.best_score['valid_0']['binary_logloss'])
    xx_tpr.append(tpr_weight_funtion(tag[val_index], valid_preds))
    xx_iteration.append(clf.best_iteration)
    xx_submit.append(clf.predict(test_x, num_iteration=clf.best_iteration))
    oof_preds[val_index] = clf.predict(train_x.iloc[val_index], num_iteration=clf.best_iteration)

print('特征个数:%d' % (len(features)))
print('线下平均LOGLOSS:%.5f' % (np.mean(xx_auc)))
print('线下全集TPR:%.5f' % (tpr_weight_funtion(tag, oof_preds)))  #线下全集TPR:0.79519
print('线下平均TPR:%.5f' % (np.mean(xx_tpr))) #线下平均TPR:0.79336
print('线下平均迭代次数:%d' % (np.mean(xx_iteration)))
print(xx_iteration)


s = 0
for i in xx_submit:
    s = s + i

test['Tag'] = list(s / 5)
submission1 = test[['UID', 'Tag']]
submission1[['UID', 'Tag']].to_csv("./submission/lgb_basesub.csv", index=False)   #0.43733