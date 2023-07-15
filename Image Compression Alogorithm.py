#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


dp=plt.imread('Spidy.png')


# In[3]:


dp


# In[4]:


plt.imshow(dp)


# In[5]:


print("shape of the image :", np.shape(dp))


# In[11]:


x_img=np.reshape(dp,(dp.shape[0]*dp.shape[1],4))


# In[12]:


np.shape(x_img)


# In[13]:


def init_cent(X,K):
    randinx=np.random.permutation(X.shape[0])
    centroid=X[randinx[:K]]
    return centroid


# In[14]:




def find_closest_centroid(X, cent):
    K = cent.shape[0]
    indx = []
    cost = 0
    for i in range(X.shape[0]):
        D = []
        for j in range(K):
            cost = np.linalg.norm(X[i] - cent[j])
            D.append(cost)
        indx.append(np.argmin(D))
    return indx


# In[15]:


import numpy as np

def compute_cent(X, K, idx):
    m, n = X.shape
    cent = np.zeros((K, n))
    for i in range(K):
        sum_val = np.zeros((n,))
        count = 0
        for j in range(X.shape[0]):
            if idx[j] == i:
                sum_val += X[j]
                count += 1
        if count != 0:
            cent[i] = sum_val / count
    return cent


# In[16]:


def run_kmeans(X,inti_cent,max_ilt):
    m,n=X.shape
    cent=inti_cent
    K = inti_cent.shape[0]
    idx=np.shape(m)
    for i in range(max_ilt):
        print("K-Means iteration %d/%d" % (i, max_ilt-1))
        idx=find_closest_centroid(X,cent)
        cent=compute_cent(X,K,idx)
    return cent,idx    


# In[26]:


K=8
max_ilter=10
initial_centroid=init_cent(x_img,K)
np.shape(initial_centroid)
cent,idx=run_kmeans(x_img,initial_centroid,max_ilter)


# In[13]:


# Find the closest centroid of each pixel
#idx = find_closest_centroid(x_img, cent)

# Replace each pixel with the color of the closest centroid
X_recovered = cent[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, dp.shape) 


# In[15]:


plt.imshow(X_recovered/225)


# In[14]:


X_recovered


# In[ ]:





# In[ ]:




