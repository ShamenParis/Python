#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


# In[107]:


transaction = pd.read_csv("Transactions.csv")
outlet = pd.read_csv("Outlet.csv")
product = pd.read_csv("Product.csv")


# In[108]:


transaction.head()


# In[119]:


Rating = transaction.groupby("OutletKey")["NetValue"].sum().reset_index()
Rating = Rating.rename(columns = {"NetValue": "Sum","OutletKey":"Key"})
df = transaction.set_index('OutletKey').join(Rating.set_index('Key'),on = "OutletKey",how="left")
#df.sort_values(by='OutletKey', ascending=False)
df = df.reset_index()
df["Rating"] = (df["NetValue"]/df["Sum"]) * 10


# In[127]:


transaction = df[["OutletKey", "ProductKey","Rating"]]


# In[128]:


transaction.head()


# In[134]:


product_new = product[["ProductKey","SKU","SKUShortName"]]
product_new = product_new[product_new["ProductKey"] != -1]


# In[135]:


df = pd.merge(transaction,product_new,on = 'ProductKey')


# In[137]:


df.head()


# In[150]:


missing_pivot = df.pivot_table(values = 'Rating',index = 'OutletKey', columns = 'SKUShortName')
missing_pivot.head()


# In[151]:


rate = {}
rows_indexes = {}
for i,row in missing_pivot.iterrows():
    rows = [x for x in range(0,len(missing_pivot.columns))]
    combine = list(zip(row.index, row.values, rows))
    rated = [(x,z) for x,y,z in combine if str(y) != 'nan']
    index = [i[1] for i in rated]
    row_names = [i[0] for i in rated]
    rows_indexes[i] = index
    rate[i] = row_names


# In[ ]:


rate


# In[153]:


pivot_table = df.pivot_table(values = 'Rating',index = 'OutletKey', columns = 'SKUShortName').fillna(0)
pivot_table = pivot_table.apply(np.sign)


# In[154]:


notrate = {}
notrate_indexes = {}
for i,row in pivot_table.iterrows():
    rows = [x for x in range(0,len(missing_pivot.columns))]
    combine = list(zip(row.index, row.values, rows))
    idx_row = [(idx,col) for idx,val,col in combine if not val > 0]
    indices = [i[1] for i in idx_row]
    row_names = [i[0] for i in idx_row]
    notrate_indexes[i] = indices
    notrate[i] = row_names


# In[ ]:





# In[172]:


n = 10
cosine_nn = NearestNeighbors(n_neighbors=n, algorithm= 'brute', metric = 'cosine')
item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values)
item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)


# In[173]:


items_dic = {}
for i in range(len(pivot_table.T.index)):
    item_idx = item_indices[i]
    col_names = pivot_table.T.index[item_idx].tolist()
    items_dic[pivot_table.T.index[i]] = col_names


# In[174]:


items_dic


# In[175]:


item_rec  = pd.DataFrame.from_dict(items_dic)


# In[178]:


item_rec = item_rec.T
item_rec.head()


# In[179]:


item_rec


# # Accuracy

# In[158]:


item_distances = 1 - item_distances


# In[160]:


predictions = item_distances.T.dot(pivot_table.T.values)/ np.array([np.abs(item_distances.T).sum(axis=1)]).T


# In[162]:


ground_truth = pivot_table.T.values[item_distances.argsort()[0]]


# In[164]:


def rmse(prediction,ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))


# In[165]:


error_rate = rmse(predictions,ground_truth)
print("Accuracy: {:.3f}".format(100 - error_rate))
print("RMSE: {:.5f}".format(error_rate))


# In[ ]:




