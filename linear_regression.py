#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports numpy, pandas, matplotlib and seaborn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


# Reads data from training data stored in csv, and cleans it
df_list = [pd.read_csv('train.csv'), pd.read_csv('test.csv')]
train_data, test_data = [df.dropna().values for df in df_list]


# In[3]:


# Does stochastic gradient descent to optimize linear coeffs, with shuffled training set
alpha, a_arr = 0.00001, np.zeros(2)
np.random.shuffle(train_data)
for x, y in train_data:
    feature_arr = np.array([1, x])
    y_hat = np.dot(a_arr, feature_arr)
    residual = y_hat - y
    a_arr -= np.multiply(alpha*residual, feature_arr)

x_arr, y_arr = train_data[:, :-1], train_data[:, -1:]
p_arr = a_arr[0] + np.multiply(a_arr[1], x_arr)


# In[4]:


# Plots training data and trained regression line
sns.set()
plt.figure(0)
plt.plot(x_arr, y_arr, 'b.')
plt.plot(x_arr, p_arr, 'r.')


# In[5]:


# Plots test data and predicted values for test points
test_x_arr, test_y_arr = test_data[:, :-1], test_data[:, -1:]
test_p_arr = a_arr[0] + np.multiply(a_arr[1], test_x_arr)

plt.figure(1)
plt.plot(test_x_arr, test_y_arr, 'b.')
plt.plot(test_x_arr, test_p_arr, 'r.')

