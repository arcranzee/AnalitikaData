#import library
import pandas as p
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#import dataset
fr = p.read_csv(r'C:\Users\arcra\OneDrive - Institut Teknologi Bandung\Kuliah\VI\TI4141 - Analitika Data\UAS\card_transdata.csv')

#mengecek dataset
fr.head(5)

#memilih variabel features dan target
feature_cols = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'used_chip', 'used_pin_number', 'online_order']
X = fr[feature_cols]
y = fr.fraud

#membagi train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#membangun model decision tree
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
clf = clf.fit(X_train, y_train)

#melakukan prediksi dari data testing
y_pred = clf.predict(X_test)

#mengevaluasi model yang telah dibangun
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#menampilkan y_test
y_test.head(5)

#menampilkan y_pred
print(y_pred)

#Membuat confusion matrix
con_mat = p.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predicted'])
sb.heatmap(con_mat, annot = True)

#menampilkan hasil prediksi
clf.predict(X_test)

#melihat data X untuk train
X_train.head(5)

#melihat data y untuk train
y_train.head(5)

#melihat data X untuk test
X_test.head(5)

#melihat data y untuk test
y_test.head(5)

#cek precision, sensitivity, dan f-score
prec = 24000/(24000+4100)
sens = 24000/(24000+2200)
f_score = (2*sens*prec)/(sens+prec)

prec
sens
f_score
