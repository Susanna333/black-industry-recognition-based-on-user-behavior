#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import gc
import networkx as nx
import time
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import deepwalk as dw

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
sns.set(style = 'white', context = 'notebook', palette = 'deep')
sns.set_style('white')

path1 = './data-firstround'
path2 = './data-secondround'

train_oper = pd.read_csv(path1 + '/operation_train_new.csv')
train_transac = pd.read_csv(path1 + '/transaction_train_new.csv')
test_oper_r1 = pd.read_csv(path1 + '/operation_round1_new.csv')
test_transac_r1 = pd.read_csv(path1 + '/transaction_round1_new.csv')

test_oper_r2 = pd.read_csv(path2 + '/test_operation_round2.csv')
test_transac_r2 = pd.read_csv(path2 + '/test_transaction_round2.csv')

oper_tran_grpuse_col = ['UID',"device1","mac1","ip1", "device_code1","device_code2","device_code3"]
oper_col = [ 'UID','ip','ip_sub','wifi']
tran_col = ['UID', 'merchant','acc_id1','acc_id2','acc_id3']

oper_tran_grpuse = pd.concat([ train_oper[oper_tran_grpuse_col],test_oper_r1[oper_tran_grpuse_col] , test_oper_r2[oper_tran_grpuse_col],                          train_transac[oper_tran_grpuse_col],test_transac_r1[oper_tran_grpuse_col],test_transac_r2[oper_tran_grpuse_col]])
oper_tran_grpuse['device_code']=oper_tran_grpuse['device_code1'].fillna(oper_tran_grpuse['device_code2'])
oper_tran_grpuse['device_code']=oper_tran_grpuse['device_code2'].fillna(oper_tran_grpuse['device_code3'])


train_oper['ip']   = train_oper['ip1'].fillna( train_oper['ip2'])
test_oper_r1['ip'] = test_oper_r1['ip1'].fillna( test_oper_r1['ip2'])
test_oper_r2['ip'] = test_oper_r2['ip1'].fillna( test_oper_r2['ip2'])

train_oper['ip_sub']   = train_oper['ip1_sub'].fillna( train_oper['ip2_sub'])
test_oper_r1['ip_sub'] = test_oper_r1['ip1_sub'].fillna( test_oper_r1['ip2_sub'])
test_oper_r2['ip_sub'] = test_oper_r2['ip1_sub'].fillna( test_oper_r2['ip2_sub'])
oper_use = pd.concat([ train_oper,test_oper_r1,test_oper_r2 ])
tran_use = pd.concat([ train_transac,test_transac_r1,test_transac_r2 ])


path3 = './edges/source/'
sourcedata = [{"usedata":oper_tran_grpuse,"useCol":oper_tran_grpuse_col},
              {"usedata":oper_use,"useCol":oper_col},
              {"usedata":tran_use,"useCol":tran_col}
             ]

def create_edgelist(useData, secNodeCol,col):
    le = LabelEncoder()
    datacp = useData[[col,secNodeCol]]
    datacp = datacp[-datacp[secNodeCol].isnull()]
    datacp = datacp[-datacp[col].isnull()]
    datacp[secNodeCol] = le.fit_transform(datacp[secNodeCol])+ 1000000
    each =datacp.groupby([col,secNodeCol])[col].agg({"trans_cnt":'count'}).reset_index()
    total = each.groupby(secNodeCol)["trans_cnt"].agg({ 'trans_cnt_total' :"sum"}).reset_index()
    gp = pd.merge(each,total,on=[secNodeCol])
    del datacp,each,total
    gp[ "ratio"] = gp['trans_cnt']/gp['trans_cnt_total']
    gp = gp.drop(['trans_cnt','trans_cnt_total'],axis = 1)
    savename = path3 +'{}_weighted_edglist_filytypeTxt.txt'.format(secNodeCol)
    np.savetxt(savename, gp.values, fmt=['%d','%d','%f'])
    gp = gp.drop("ratio",axis = 1)
    savenameForDeepWalk = path3 + '{}_weighted_edglist_DeepWalk.txt'.format(secNodeCol)
    np.savetxt(savenameForDeepWalk, gp.values, fmt=['%d','%d'])
    del gp

# 一度关联
for spec in sourcedata:
    for c in spec["useCol"]:
        if c != "UID":
            create_edgelist(spec["usedata"], c, 'UID','')
            gc.collect()


def create_edgelist2(useData, secNodeCol,col,degreename):
    le = LabelEncoder()
    datacp = useData[['UID',col,secNodeCol]]
    datacp = datacp[-datacp[secNodeCol].isnull()]
    datacp = datacp[-datacp[col].isnull()]
    datacp[secNodeCol] = le.fit_transform(datacp[secNodeCol])+ 1000000
    datacp[col] = le.fit_transform(datacp[col])+ 10000000
    each =datacp.groupby(['UID',col,secNodeCol])[col].agg({"trans_cnt":'count'}).reset_index()
    total = each.groupby(secNodeCol)["trans_cnt"].agg({ 'trans_cnt_total' :"sum"}).reset_index()
    gp = pd.merge(each,total,on=[secNodeCol])
    del datacp,each,total
    gp[ "ratio"] = gp['trans_cnt']/gp['trans_cnt_total']
    temp=gp.groupby(col)['trans_cnt'].agg({'cnt2':'sum'}).reset_index()
    gp = gp.merge(temp,on=[col],how='left')
    gp['ratio2'] = gp['trans_cnt']/gp['cnt2']
    gp = gp.drop(['trans_cnt','trans_cnt_total','cnt2'],axis = 1)
    savename = path4 + degreename+ '{}_{}_weighted_edglist_filytypeTxt.txt'.format(secNodeCol,col)
    np.savetxt(savename, gp.values, fmt=['%d','%d','%d','%f','%f'])
    del gp

path4='./edges/2度关联/'
#二度关联
for x in ['acc_id2','acc_id3']:
    create_edgelist2(tran_use, x, 'acc_id1','2degree')
    
for x in ['ip1','device_code','device1']:
    create_edgelist2(oper_tran_grpuse, x, 'mac1','2degree')


def createEdgeFomat(fname):
    G = nx.Graph()
    f = open(fname,'r')
    lines = f.readlines()
    f.close()
    lines =[line.replace("\n","").split(" ")  for line in lines]
    lines = [[int(x[0]),int(x[1]),float(x[2])] for x in lines]
    edfname = fname.replace(".txt",".edgelist")
    
    for edg in lines:
        G.add_edge(edg[0], edg[1], weight=edg[2])
    print("\n-------------------------------------\n")
    print("saving fali name %s " % edfname)
    print("\n-------------------------------------\n")
    fh=open(edfname,'wb')
    nx.write_edgelist(G, fh)
    fh.close()

for f in os.listdir(path3):
    if "DeepWalk" not in f:
        print("creating %s edge format for node2vec embedding ... " % (f.replace( "_edglist_filytypeTxt.txt", "" )) )
        createEdgeFomat(path3 + f)
        print(f.split(".")[0],"finish")


def createEdgeFomat2(fname):
    G = nx.Graph()
    with open(fname,'r') as ff:
        lines = ff.readlines()
    lines =[line.replace("\n","").split(" ")  for line in lines]
    lines = [[int(x[0]),int(x[1]),int(x[2]),float(x[3]),float(x[4])] for x in lines]
    edfname = fname.replace(".txt",".edgelist")
    
    for edg in lines:
        G.add_edge(edg[0], edg[1], weight=edg[4])
        G.add_edge(edg[1], edg[2], weight=edg[3])
    print("\n-------------------------------------\n")
    print("saving fali name %s " % edfname)
    print("\n-------------------------------------\n")
    with open(edfname,'wb') as fh:
        nx.write_edgelist(G, fh)

path4='./edges/2度关联/'
for f in os.listdir(path4):
    print("creating %s edge format for node2vec embedding ... " % (f.replace( "_edglist_filytypeTxt.txt", "" )) )
    createEdgeFomat2(path4 + f)
    print(f.split(".")[0],"finish")

####node2vec
import networkx as nx
from node2vec import Node2Vec
import sys

def emb_graph_2vec(inputpath,dim):
    print("input name will be ",inputpath)
    emb_name = inputpath.replace("weighted_edglist_filytypeTxt.edgelist","")
    print("emb_name will be ",emb_name)

    savename =inputpath.replace("weighted_edglist_filytypeTxt.edgelist",".emb")
    print("emb outfile name will be ",savename)
    if os.path.exists(savename):
        print("file alread exists in cache, please rename")
        sys.exit(1)

    graph = nx.read_edgelist(inputpath,create_using=nx.DiGraph())
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, workers=10) 
    # Embed nodes
    print("training .... ")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print("training finished saving result... ")

    print("saving %s file to disk "%savename)
    # Save embeddings for later use
    model.wv.save_word2vec_format(savename)
    print("done")
    # Save model for later use


import os
dataRoot =  './edges/source/'
inputpath = dataRoot + "mac1_weighted_edglist_filytypeTxt.edgelist"
try:
    emb_graph_2vec(inputpath,36)
except Exception as e:
    print(e)


dataRoot =  './edges/source/'
inputpath =  dataRoot +  "merchant_weighted_edglist_filytypeTxt.edgelist"
try:
    emb_graph_2vec(inputpath,64)
except Exception as e:
    print(e)


dataRoot =  './edges/source/'
inputpath =  dataRoot +  "acc_id1_weighted_edglist_filytypeTxt.edgelist"
try:
    emb_graph_2vec(inputpath,36)
except Exception as e:
    print(e)

dataRoot =  './edges/2度关联/'
filenames=[
    '2degreeacc_id2_acc_id1_weighted_edglist_filytypeTxt.edgelist',
    '2degreeacc_id3_acc_id1_weighted_edglist_filytypeTxt.edgelist',
    '2degreedevice_code_mac1_weighted_edglist_filytypeTxt.edgelist',
    '2degreedevice1_mac1_weighted_edglist_filytypeTxt.edgelist',
    '2degreeip1_mac1_weighted_edglist_filytypeTxt.edgelist'
]

for filename in filenames:
    inputpath =  dataRoot +  filename
    try:
        emb_graph_2vec(inputpath,36)
    except Exception as e:
        print(e)

