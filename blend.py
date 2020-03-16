#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
pre1 = pd.read_csv('./submission/lgb_basesub_0.43633.csv').rename(columns={'Tag': 'Tag1'})
pre2 = pd.read_csv('./submission/lgbemb1.csv').rename(columns={'Tag': 'Tag2'})
pre3 = pd.read_csv('./submission/lgbemb2.csv').rename(columns={'Tag': 'Tag3'})

pre = pre1.merge(pre2, on=['UID'], how='left')
pre = pre.merge(pre3, on=['UID'], how='left')
pre['Tag'] = 0.50*pre['Tag1']+0.25*pre['Tag2']+0.25*pre['Tag3']

pre[['UID', 'Tag']].to_csv('./submission/blend.csv', index=False)


# In[ ]:




