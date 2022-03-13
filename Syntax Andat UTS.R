#CASE 2
#CLASSIFICATION WITH DECISION TREE

#Lakukan import dataset
library(readxl)
Data_UTS <- read_excel("C:/Users/arcra/OneDrive - Institut Teknologi Bandung/Kuliah/VI/TI4141 - Analitika Data/Data UTS.xlsx")

#Periksa data-data dari dataset "Data_UTS" (nama kolom, dll)
summary(Data_UTS)
str(Data_UTS)

#Lakukan import package untuk cleaning data
#1. Import package "dplyr"
install.packages("dplyr")
library(dplyr)

#2. Import package "plyr"
install.packages("plyr")
library(plyr)

#3. Import package "DataExplorer"
install.packages("DataExplorer")
library(DataExplorer)

#DATA CLEANING
#Cleaning data-data yang terduplikasi
Data_UTS <- distinct(Data_UTS, .keep_all=FALSE)

#melihat jumlah missing data
plot_missing(Data_UTS)

#Mengganti kolom yang kosong dengan nilai NA
Data_UTS$WatchDuration[which(is.na(Data_UTS$WatchDuration),arr.ind=TRUE)] <- mean(na.omit(Data_UTS$WatchDuration))
Data_UTS$VidRating[which(is.na(Data_UTS$VidRating),arr.ind=TRUE)] <- mean(na.omit(Data_UTS$VidRating))
Data_UTS$ClassRating[which(is.na(Data_UTS$ClassRating),arr.ind=TRUE)] <- mean(na.omit(Data_UTS$ClassRating))

                                                                                  
#Menghapus kolom yang tidak diperlukan
#Penghapusan kolom "UserID" karena dalam membuat model, pelanggan tidak perlu dilihat secara individual
Data_UTS$UserID <- NULL

#Lakukan pembagian (split) data menjadi train dan test data
#Train data akan digunakan untuk membangun model sedangkan test data akan digunakan untuk melakukan evaluasi model
#Komposisi train data adalah 70% dan komposisi test data adalah 30%

#set seed sebesar 56 untuk men-generate random number yang sama setiap melakukan running model
set.seed(56)

#Buat pembagian/partisi data pada kolom churn sesuai komposisi yang telah ditentukan diatas
#Import package "caret" untuk menggunakan fungsi createDataPartition
install.packages("caret")
library("caret")

#Buat pembagian data dengan fungsi createDataPartition
#createDataPartition(kolom yang menentukan partisi, persentase data yang masuk ke partisi, logika pembentukan tipe data list)
split <- createDataPartition(Data_UTS$Churn, p = 0.7, list=FALSE)

#Buat train data dan test data
#test/train data <- Telecom_Data[partisi data untuk test/train,]
dtrain <- Data_UTS[split,]
dtest <- Data_UTS[-split,]

#Lakukan import package untuk decision tree
#1. Import package "rpart"
install.packages("rpart")
library(rpart)

#2. Import package "rpart.plot"
install.packages("rpart.plot")
library(rpart.plot)

#Buat decision tree prediksi churn dengan prediktor semua kolom pada dataset Telecom_Churn
decision_tree <- rpart(Churn ~., data=Data_UTS, method="class")

#Plot decision tree
rpart.plot(decision_tree)

#Lakukan evaluasi terhadap model decision tree yang telah dibuat sebelumnya dengan dataset dtest

#Evaluasi dengan confusion matrix
#Mencari true positive (TP), true negative (TN), false positive(FP), dan false negative (FN)

#Gunakan fungsi predict untuk memasukkan dtest ke model dan buat dataset baru berisi hasil fungsi predict
#predict(model, test data)
pob_pred <- predict(decision_tree, dtest)

#Lihat hasil fungsi predict
pob_pred

#Tentukan peluang keputusan seseorang dikatakan churn dan tidak berdasarkan hasil predict
#Jika peluang churn lebih besar dari 50% maka orang tersebut dikatakan churn (yes)

#Buat dataset baru yang berisi hasil keputusan churn berdasarkan dataset hasil fungsi predict
#Gunakan logika ifelse memudahkan pengisian
#ifelse(condition, value if true, value if false)
tree_Pred <- ifelse(pob_pred[,2]>0.5,"Yes","No")

#Gunakan fungsi tabel untuk melihat nilai TP, TN, FP, dan FN dengan melihat frekuensi "yes" dan "no" untuk churn di dtree_pred1 dibandingkan dengan data aktual churn dari dtest
#table(nama dataset 1, nama dataset 2)
confusion_matrix <- table(Predicted=tree_Pred, Actual=dtest$Churn)

#Menghitung akurasi
#Total keputusan yang benar/total keputusan yang salah
accuracy <- sum(diag(confusion_matrix))/sum(confusion_matrix)

