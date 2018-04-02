
# coding: utf-8

# In[1]:


#get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt

import datasets.data_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import math
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
# # Part 1

# In[3]:

#Fast DTW implementation with window w as from the paper referenced in Readme point2
def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

'''Following function creates a lower bound distance which helps to avoid DTW calculation in many cases.
 Again its idea has been taken from paper refernced in Readme point2'''
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)


'''1-nn implemenation for all the points with DTW as distance metric'''
def knn(train,test,w):
    preds=[]
    ic = 0
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        jc = 0 
        #print ind
        for j in train:
            print(ic,jc)
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
            jc = jc +1
        preds.append(closest_seq[-1])
        ic = ic+1
    return classification_report(test[:,-1],preds)
# Load dataset
x, y = datasets.data_reader.read_clean_dataset(summary=True)

# TODO: 
# 1. Analyze the dataset (open-ended). 
# 2. Train a classifier on the dataset and report the results. 

# x = StandardScaler().fit_transform(x)
# pca = PCA(0.9)
# trainx = x[:25000]
# trainy = y[:25000]
# testx = x[25000:]
# testy = y[25000:]
# trainx = pca.fit_transform(trainx)
# testx = pca.transform(testx)


trainy = trainy.reshape((25000,1))
testy = testy.reshape((5000,1))


train = np.concatenate((trainx, trainy), axis = 1)
test =  np.concatenate((testx, testy), axis = 1)


print(knn(train, test, 5))
# #print (pca.explained_variance_ratio_)

# logisticRegr = LogisticRegression(solver = 'lbfgs')
# logisticRegr.fit(trainx, trainy)
# pred = logisticRegr.predict(testx)
# np.savetxt('train.txt', pred)
# np.savetxt('testy.txt', testy)
# print(logisticRegr.score(testx, testy))
# # # Part 2

# # In[3]:

# z = []
# one = []
# x, x_len = datasets.data_reader.read_corrupted_dataset(summary=True)
# for i in range(457):
#     if x[0,i] > 0:
#         z.append(i)
#     if x[1,i] > 0:
#         one.append(i)

# print(z)
# print(one)
# x = pca.transform(x)
# print(x.shape)
# pred = logisticRegr.predict(x)
# print(pred)
# print(pred.shape)
# np.savetxt('label.txt', pred)
# TODO: 
# 1. Classify each data point. 
# 2. Find the optimal alignment of each data point. 

