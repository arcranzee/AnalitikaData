#!/usr/bin/env python
# coding: utf-8

# In[42]:


#import pandas
import pandas as p


# In[43]:


#menyiapkan data frame
ds = p.DataFrame(
    {"a" : [4 ,5, 6],
    "b" : [7, 8, 9],
    "c" : [10, 11, 12]},
        index = [1, 2, 3])

#menampilkan dataframe
print(ds)


# In[44]:


#menyiapkan data frame
df = p.DataFrame([[3, 4], [5, 6]])

#menambahkan baris baru
a_row = p.Series([1, 2])
row_df = p.DataFrame([a_row])

#menggabungkan baris baru ke dataset
df = p.concat([row_df, df])

#menampilkan dataframe
print(df)


# In[45]:


#nama file Excel, digunakan di banyak fungsi
excel_file = 'movies.xls'

#membuat dataframe
movies = p.read_excel(excel_file)


# In[46]:


#menampilkan data teratas (defaultnya 5)
movies.head()

#menampilkan data terbawah (defaultnya 5)
movies.tail()

#mengecek ukuran dataset (deafultnya hanya untuk sheet pertama)
movies.shape

#mengecek data dari baris spesifik
movies.loc[186]


# In[47]:


#mengimport sheet spesifik
movies_1 = p.read_excel(excel_file, sheet_name=0, index_col=0)
movies_2 = p.read_excel(excel_file, sheet_name=1, index_col=0)
movies_3 = p.read_excel(excel_file, sheet_name=2, index_col=0)

#menampilkan sheet spesifik
movies_1.shape
movies_3.shape
movies_2.shape


# In[48]:


#menggabungkan tiga buah dataset
movies_all = p.concat([movies_1, movies_2, movies_3])

#mengecek ukuran dataset gabungan
movies_all.shape


# In[49]:


#mengimpor banyak sheet ke satu dataset
#dapat menggunakan nama file Excel atau excel_file yang sudah didefinisikan di awal
xlsx = p.ExcelFile(excel_file)

#menggabungkan seluruh sheet yang baru diimport
movies_sheets = []
for sheet in xlsx.sheet_names:
   movies_sheets.append(xlsx.parse(sheet))
movies_all2 = p.concat(movies_sheets)

#mengecek ukuran dataset baru
movies_all2.shape


# In[50]:


#mengekspor dataframe/dataset ke dalam file Excel di folder yang sama
movies_all2.to_excel("Film.xlsx")


# In[51]:


#pengurutan berdasarkan pendapatan kotor
movies_sort_gross = movies.sort_values(['Gross Earnings'], ascending=False)

#menampilkan dataset yang sudah diurutkan
movies_sort_gross.head()


# In[52]:


#import library untuk membuat plot
import matplotlib.pyplot as plt


# In[53]:


#membuat bar chart untuk total pendapatan kotor
movies_sort_gross['Gross Earnings'].head(11).plot(kind="barh")
plt.show()

#membuat histogram untuk rata-rata skor IMDB
movies['IMDB Score'].plot(kind="hist")
plt.show()


# In[54]:


#menampilkan statistik dasar
movies_all2.describe()

#melakukan operasi pada data
movies_all2['Net Earnings'] = movies_all2['Gross Earnings'] - movies_all2['Budget']

#menampilkan statistik secara sepesifik
movies_all2['Net Earnings'].mean()


# In[55]:


#import numpy
import numpy as np


# In[56]:


#membuat dataframe baru
df_new = p.read_excel("movies.xls")

#menghitung jumlah film yang tidak memiliki pendapatan kotor
df_new['Gross Earnings'].isnull().sum()

#menghapus kolom
to_drop = ['Country']
df_new.drop(to_drop, inplace=True, axis=1)

#menampilkan dataset yang sudah didrop
df_new.head()

