#import library
import pandas as p
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#import dataset
fr = p.read_csv(r'C:\Users\arcra\OneDrive - Institut Teknologi Bandung\Kuliah\VI\TI4141 - Analitika Data\UAS\card_transdata.csv')

#mengecek dataset
fr.head(5)

#mengecek struktur dataset
fr.shape

#mengecek data yang kosong
fr.isnull()
fr.isna().sum()

#Pembuatan dataframe tiap fitur untuk proses perhitungan korelasi
d1 = fr[['distance_from_home', 'fraud']]
d2 = fr[['distance_from_last_transaction', 'fraud']]
d3 = fr[['ratio_to_median_purchase_price', 'fraud']]
d4 = fr[['repeat_retailer', 'fraud']]
d5 = fr[['used_chip', 'fraud']]
d6 = fr[['used_pin_number', 'fraud']]
d7 = fr[['online_order', 'fraud']]

#Perhitungan korelasi dataframe 1
d1.corr()

#Perhitungan korelasi dataframe 2
d2.corr()

#Perhitungan korelasi dataframe 3
d3.corr()

#Perhitungan korelasi dataframe 4
d4.corr()

#Perhitungan korelasi dataframe 5
d5.corr()

#Perhitungan korelasi dataframe 6
d6.corr()

#Perhitungan korelasi dataframe 7
d7.corr()

#membuat dataset baru dari dataset yang sudah dibersihkan
fr_new = fr.drop(labels = 'repeat_retailer', axis=1)

#mengecek dataset baru
fr_new.head(5)

#mengecek struktur dataset baru
fr_new.shape

#assign tiap fitur pada variabel dan parameter
X = fr_new[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'used_chip', 'used_pin_number', 'online_order']]
y = fr_new['fraud']

#membuat train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

#membangun model regresi logistik
logit = LogisticRegression()

#membuat model dari dataset training sebanyak 25%
logit.fit(X_train, y_train)

#mengaplikasikan 25% dataset test dari model yang terbangun
y_pred = logit.predict(X_test)

#mengecek atribut model
logit.classes_

#mengecek nilai koefisien intersep (b_0) dan slope (b_1, b_2, b_3, b_4, b_5, b_6) dari model
logit.intercept_
logit.coef_

#mengecek akurasi model
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()

#mengecek akurasi model yang ditest dengan cara lain
logit.score(X_test,y_test)

#melakukan fitting X_test ke model yang sudah terbentuk (y_pred)
logit.predict(X_test)

#membuat confusion matrix
con_mat = p.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
sb.heatmap(con_mat, annot=True)

#membandingkan nilai y_test (dari dataset awal) dengan y_pred
print(y_pred)
y_test.head(5)

#melihat data X untuk train
X_train.head(5)

#melihat data y untuk train
y_train.head(5)

#melihat data X untuk test
X_test.head(5)

#membuat tabel korelasi
corr = fr.corr()
corr

#melakukan visualisasi terhadap tabel korelasi
f, ax = plt.subplots(figsize =(9, 8)) 
sb.heatmap(corr,ax=ax, annot=True, cmap = 'Greens',linewidths=0.5)

#mengecek akurasi data train
logit.score(X_train, y_train)

#Distribution of Cases with Yes/No Fraud according to Online Orders
fig, ax = plt.subplots(figsize = (13,6))
ax.hist(fr[fr["fraud"]==1]["online_order"], bins=15, alpha=0.5, color="red", label="Fraud")
ax.hist(fr[fr["fraud"]==0]["online_order"], bins=15, alpha=0.5, color="#fccc79", label="No Fraud")
ax.set_xlabel("Online Orders")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of Cases with Yes/No Fraud according to Online Orders")
ax.legend();

#Distribution of Cases with Yes/No Fraud according to Used Pin Number
fig, ax = plt.subplots(figsize = (13,6))
ax.hist(fr[fr["fraud"]==1]["used_pin_number"], bins=15, alpha=0.5, color="red", label="Fraud")
ax.hist(fr[fr["fraud"]==0]["used_pin_number"], bins=15, alpha=0.5, color="#fccc79", label="No Fraud")
ax.set_xlabel("Used Pin Number")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of Cases with Yes/No Fraud according to Used Pin Number")
ax.legend();

#Distribution of Cases with Yes/No Fraud according to Used Chip
fig, ax = plt.subplots(figsize = (13,6))
ax.hist(fr[fr["fraud"]==1]["used_chip"], bins=15, alpha=0.5, color="red", label="Fraud")
ax.hist(fr[fr["fraud"]==0]["used_chip"], bins=15, alpha=0.5, color="#fccc79", label="No Fraud")
ax.set_xlabel("Used Chip")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of Cases with Yes/No Fraud according to Used Chip")
ax.legend();

#Distribution of correlation of features
sb.set_style('white')
sb.set_palette('Paired')
plt.figure(figsize = (13,6))
plt.title('Distribution of correlation of features')
abs(corr['fraud']).sort_values()[:-1].plot.barh()
plt.show()

#Distribution of Distance From Home
fig, ax = plt.subplots(figsize = (10,5))
sb.kdeplot(fr[fr["fraud"]==1]["distance_from_home"], alpha=0.5,shade = True, color="red", label="Fraud", ax = ax)
sb.kdeplot(fr[fr["fraud"]==0]["distance_from_home"], alpha=0.5,shade = True, color="#fccc79", label="No Fraud", ax = ax)
plt.title('Distribution of Distance From Home', fontsize = 18)
ax.set_xlabel("Distance From Home")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

#Distance From Last Transaction
fig, ax = plt.subplots(figsize = (13,5))
sb.kdeplot(fr[fr["fraud"]==1]["distance_from_last_transaction"], alpha=0.5,shade = True, color="red", label="Fraud", ax = ax)
sb.kdeplot(fr[fr["fraud"]==0]["distance_from_last_transaction"], alpha=0.5,shade = True, color="#fccc79", label="No Fraud", ax = ax)
plt.title('Distribution of Distance From Last Transaction', fontsize = 18)
ax.set_xlabel("Distance From Last Transaction")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()
