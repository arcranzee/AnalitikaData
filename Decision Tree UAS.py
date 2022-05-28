#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import library
import pandas as p
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#import dataset
fr = p.read_csv(r'C:\Users\arcra\OneDrive - Institut Teknologi Bandung\Kuliah\VI\TI4141 - Analitika Data\UAS\card_transdata.csv')


# In[4]:


#mengecek dataset
fr.head(5)


# In[5]:


#memilih variabel features dan target
feature_cols = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'used_chip', 'used_pin_number', 'online_order']


# In[6]:


X = fr[feature_cols]
y = fr.fraud


# In[7]:


#membagi train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[8]:


#membangun model decision tree
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
clf = clf.fit(X_train, y_train)


# In[9]:


#melakukan prediksi dari data testing
y_pred = clf.predict(X_test)


# In[10]:


#mengevaluasi model yang telah dibangun
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[19]:


#menampilkan y_test
y_test.head(5)


# In[12]:


#menampilkan y_pred
print(y_pred)


# In[13]:


#Membuat confusion matrix
con_mat = p.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
sb.heatmap(con_mat, annot = True)


# In[14]:


#menampilkan hasil prediksi
clf.predict(X_test)


# In[15]:


#melihat data X untuk train
X_train.head(5)


# In[16]:


#melihat data y untuk train
y_train.head(5)


# In[17]:


#melihat data X untuk test
X_test.head(5)


# In[18]:


#melihat data y untuk test
y_test.head(5)

