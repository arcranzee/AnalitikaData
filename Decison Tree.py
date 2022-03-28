#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import packages
import pandas as p
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sb


# In[7]:


#membuat nama kolom
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']


# In[8]:


#menambahkan dataset dari csv ke nama kolom yang sudah dibuat
pima = p.read_csv("pima-indians-diabetes.csv", header = None, names = col_names) #dataset yang digunakan


# In[9]:


#menampilkan 5 abris pertama
pima.head(5)


# In[10]:


#melihat ukuran dataset
pima.shape


# In[11]:


#memilih variabel features dan target
feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']

X = pima[feature_cols]
y = pima.label


# In[12]:


#membagi train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[13]:


#membangun model decision tree
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
clf = clf.fit(X_train, y_train)


# In[14]:


#melakukan prediksi dari data testing
y_pred = clf.predict(X_test)


# In[15]:


#mengevaluasi model yang telah dibangun
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[18]:


#menampilkan y_test
y_test.head(15)


# In[19]:


#menampilkan y_pred
print(y_pred)


# In[20]:


#Membuat confusion matrix
con_mat = p.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
sb.heatmap(con_mat, annot = True)


# In[21]:


#menampilkan hasil prediksi
clf.predict(X_test)


# In[31]:


#menambahkan data abru
new_candidates = {'pregnant': [6,1,10,0],
                  'glucose': [200,90,210,85],
                  'bp': [72,60,75,65],
                  'insulin': [90,10,100,15],
                  'bmi': [35,23,33,24],
                  'pedigree': [2.5,0.7,2.2,0.8],
                  'age': [55,25,45,27],
                  }

df2 = p.DataFrame(new_candidates, columns= ['pregnant', 'glucose', 'bp', 'insulin','bmi', 'pedigree', 'age'])


# In[32]:


#mengecek data baru
print(df2)


# In[33]:


#melakukan prediksi terhadap data baru
y_pred_new = clf.predict(df2)
print(y_pred_new)


# In[ ]:




