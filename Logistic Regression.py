#!/usr/bin/env python
# coding: utf-8

# In[11]:


#import packages
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


# In[12]:


#membuat dataset
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

df = p.DataFrame(candidates,columns = ['gmat', 'gpa','work_experience','admitted'])


# In[13]:


#membuat array independent (x) dan dependent (y) variables
#x1 (GMAT), x2 (GPA), x3 (work experience)
X = df[['gmat', 'gpa', 'work_experience']]
y = df['admitted']


# In[14]:


#membuat train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


# In[15]:


#membangun model model regresi logistik
#LogisticRegression() dapat didefinisikan solvernya, misal 'liblinear'
logit = LogisticRegression()


# In[16]:


#membuat model dari dataset training sebanyak 25%
logit.fit(X_train, y_train)


# In[17]:


#mengaplikasikan 25% dataset test dari model yang terbangun
y_pred = logit.predict(X_test)


# In[18]:


#mengecek atribut model
logit.classes_


# In[19]:


#mengecek nilai koefisien intersep (b_0) dan slope (b_1, b_2, b_3) dari model
logit.intercept_
logit.coef_


# In[20]:


#mengecek akurasi model
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[21]:


#mengecek akurasi model dengan cara lain
logit.score(X_test,y_test)


# In[22]:


#melakukan fitting X_test ke model yang sudah terbentuk (y_pred)
logit.predict(X_test)


# In[23]:


#membandingkan nilai y_test (dari dataset awal) dengan y_pred
print(y_test)


# In[24]:


#membuat confusion matrix
con_mat = p.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
sb.heatmap(con_mat, annot=True)


# In[26]:


#menambahkan data baru
new_candidates = {'gmat': [590,740,680,610,710],
                  'gpa': [2,3.7,3.3,2.3,3],
                  'work_experience': [3,4,6,1,5]
                  }

df2 = p.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])


# In[27]:


#mencoba klasifikasi data baru pada model
y_pred_new = logit.predict(df2)


# In[28]:


#mengecek klasifikasi model pada data baru
print(y_pred_new)

