#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import random
from sklearn.decomposition import PCA
warnings.simplefilter(action='ignore', category=FutureWarning)
dataroot = './data/'
cacheRoot = './edges/source/'
path1 = './data-firstround'
path2 = './data-secondround'
data =  pd.read_csv(dataroot + "groupdata.csv")
tag_train = pd.read_csv(path1 + '/tag_train_new.csv')
submission = pd.read_csv(path2+'./submit_example.csv')
tagdata = pd.concat([tag_train, submission], axis=0, ignore_index=True)

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=0)


def get_embeding(fname, embname):
    with open(fname,'r') as f:
        embeding_lines = f.readlines()
    mapfunc = lambda x: list(map(float,x))
    embeding_lines = [li.replace("\n","").split(" ") for li in embeding_lines[1:]]
    embeding_lines = [[int(line[0])] + mapfunc(line[1:]) for line in embeding_lines]
    cols = ["UID"] + [embname + str(i) for i in range(len(embeding_lines[0])-1)]
    embeding_df = pd.DataFrame(embeding_lines, columns=cols )
    del embeding_lines
    return embeding_df


mac1_emb = get_embeding(cacheRoot + "mac1_.emb", "mac1_emb_")
merchant_emb = get_embeding(cacheRoot + "merchant_.emb", "merchant_emb_")
# merchant_dbk = get_embeding(cacheRoot + "merchant_weighted_edglist_DeepWalk.embeddings", "merchant_deepwalk_")
mac1_emb = mac1_emb[ mac1_emb.UID.map(lambda x: x<= 131587)]
merchant_emb = merchant_emb[ merchant_emb.UID.map(lambda x: x<= 131587)] 
# merchant_dbk = merchant_dbk[ merchant_dbk.UID.map(lambda x: x<= 131587)]

mac_merch = pd.DataFrame()
mac_merch['UID'] = list(set(merchant_emb.UID.tolist() +  mac1_emb.UID.tolist()))
mac_merch = mac_merch.merge( mac1_emb, on = ["UID"],how = "left"  )
mac_merch = mac_merch.merge( merchant_emb, on = ["UID"],how = "left"  )
mac_merch = mac_merch.fillna( 0.0 )

#降维
dim = 42
pca = PCA(n_components=dim)
pca_res = pca.fit_transform(mac_merch.drop("UID", axis = 1))
me_pca = pd.DataFrame(pca_res, columns=["pca_mac1_merchrant_%d" % i for i in range(dim)])
me_pca["UID"] = mac_merch.UID.values

datapca = pd.merge(data,me_pca,on = ['UID'],how = 'left')
datapca = tagdata.merge(datapca,on = ['UID'],how = 'left')
data = data.merge(mac1_emb, on = ['UID'],how = 'left')
data = data.merge(merchant_emb, on = ['UID'],how = 'left')
data = tagdata.merge(data,on = ['UID'],how = 'left')

def  process_feature(train_x, valid_x, test_df):
    result = []
    for df in [train_x, valid_x, test_df]:
        result.append(df.drop('Tag', axis=1))
    return result 
def cv(df, num_folds, param, stratified=True, debug=False):
    train_df = df[df.Tag != 0.5]
    test_df = df[df.Tag == 0.5]
    
    seed = 0
    if "seed" in param: 
        seed = param["seed"]
        del param['seed']
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(train_df.shape[0])
    all_test_preds = []    
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['Tag'])):
        train_x, train_y = train_df.iloc[train_idx], train_df['Tag'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_df['Tag'].iloc[valid_idx]
        fold_preds = test_df[["UID"]]
        
        train_x, valid_x, test = process_feature(train_x, valid_x, test_df)
        if n_fold == 0:
            print(train_x.shape, valid_x.shape, test.shape)
        
        train_data = lgb.Dataset(train_x, label=train_y)
        validation_data = lgb.Dataset(valid_x, label=valid_y)

        clf=lgb.train(param,
                      train_data,
                      num_boost_round=10000,
                      valid_sets=[train_data, validation_data],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=100,
                      verbose_eval=100)

        valid_preds = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_preds = clf.predict(test, num_iteration=clf.best_iteration)

        fold_preds['Tag'] = test_preds
        fold_preds['fold_id'] = n_fold + 1
        all_test_preds.append(fold_preds)
        
        oof_preds[valid_idx] = valid_preds
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, valid_preds)))
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    full_auc = roc_auc_score(train_df['Tag'], oof_preds)
    all_test_preds = pd.concat(all_test_preds, axis=0)
    sub = pd.DataFrame()
    sub['UID'] = all_test_preds.UID.unique()
    sub.set_index("UID", inplace=True)
    sub["Tag"] = all_test_preds.groupby("UID").Tag.mean()
    print('Full AUC score %.6f' % full_auc)
    
    return [full_auc,sub]

def runmodel(temp):
    params_list =[
     {'boosting_type': 'goss', 'colsample_bytree': 0.6555004575000242, 'learning_rate': 0.016380004820033073, 'max_bin': 1000, 'metric': 'auc', 'min_child_weight': 2.1115838168176433, 'num_leaves': 108, 'reg_alpha': 23.247001339889128, 'reg_lambda': 997.9576062039534, 'subsample': 1.0, 'verbose': 1, 'seed': 4419},
     {'boosting_type': 'goss', 'colsample_bytree': 0.7187703092392053, 'learning_rate': 0.01939219215282862, 'max_bin': 310, 'metric': 'auc', 'min_child_weight': 2.66983907940641, 'num_leaves': 89, 'reg_alpha': 18.48224434106526, 'reg_lambda': 470.54675380054465, 'subsample': 1.0, 'verbose': 1, 'seed': 57},
     {'boosting_type': 'goss', 'colsample_bytree': 0.6274617979582582, 'learning_rate': 0.01680918441243103, 'max_bin': 780, 'metric': 'auc', 'min_child_weight': 0.8226071606806127, 'num_leaves': 73, 'reg_alpha': 14.466924422050258, 'reg_lambda': 658.5772060624658, 'subsample': 1.0, 'verbose': 1, 'seed': 1732},
     {'boosting_type': 'goss', 'colsample_bytree': 0.6991838451153098, 'learning_rate': 0.01577419276366034, 'max_bin': 940, 'metric': 'auc', 'min_child_weight': 7.758954855388241, 'num_leaves': 187, 'reg_alpha': 43.60868666926589, 'reg_lambda': 667.6371302027073, 'subsample': 1.0, 'verbose': 1, 'seed': 139},
     {'boosting_type': 'goss', 'colsample_bytree': 0.6198406436401038, 'learning_rate': 0.016662891953242748, 'max_bin': 820, 'metric': 'auc', 'min_child_weight': 3.5602833459924015, 'num_leaves': 77, 'reg_alpha': 13.398041512170746, 'reg_lambda': 631.8105595021391, 'subsample': 1.0, 'verbose': 1, 'seed': 7609}
    ]
    result = []
    for params in params_list:
        result.append(cv(temp,5,params))
    return result


sub_embeding = runmodel(data)
def zerone(x):
    if x < 0:
        x = 0
    elif x>1:
        x = 1
    return x

all_subs = [x[1] for x in sub_embeding]
all_sub3 = pd.concat(all_subs)
all_sub3 = all_sub3.groupby("UID")["Tag"].agg({"Tag":"mean" }).reset_index()
all_sub3.Tag = all_sub3.Tag * 2.5
all_sub3['Tag'] = all_sub3.apply(lambda x:zerone(x['Tag']),axis=1)
subRoot = './submission/'
all_sub3.to_csv(subRoot + "lgbemb1.csv", encoding = 'utf8', index = False)


# In[73]:


sub_embedingpca = runmodel(datapca)
all_subspca = [x[1] for x in sub_embedingpca]
all_subspca = pd.concat(all_subs)
all_subspca = all_subspca.groupby("UID")["Tag"].agg({"Tag":"mean" }).reset_index()
all_subspca['Tag'] = all_subspca.apply(lambda x:zerone(x['Tag']),axis=1)
all_subspca.to_csv(subRoot + "lgbemb2.csv", encoding = 'utf8', index = False)
