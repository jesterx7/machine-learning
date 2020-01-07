from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

# Inialisasi Fungsi untuk Model KNN
def KNN_Model(x_train, y_train, k):
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(x_train, y_train)

	return knn

# Fungsi untuk melakukan K-Fold Cross Validation
def KFoldCV(data_x, data_y, n_split):
	kf = KFold(n_splits = n_split)
	kf.get_n_splits(data_x)

	train_x, test_x, train_y, test_y = ([] for i in range(4))
	for n, (train_index, test_index) in enumerate(kf.split(data_x)):
		train_x.append(data_x[train_index])
		test_x.append(data_x[test_index])
		train_y.append(data_y[train_index])
		test_y.append(data_y[test_index])

	train_data = {'x':train_x, 'y':train_y}
	test_data = {'x':test_x, 'y':test_y}

	return train_data, test_data

# Inialisasi dataset yang akan dipakai
datasets = load_iris()
data_train = datasets.data
data_labels = datasets.target

# Inialisasi jumlah n_split (jumlah fold / grup) dari K-Fold Cross Validation
# Dan kemudian dataset tadi displit (dipisah) menggunakan fungsi K-Fold Cross Validation tadi
split = 5
train, test = KFoldCV(data_train, data_labels, split)

# Inialisasi variable scores yang akan digunakan untuk menyimpan semua akurasi dari setiap grup / folds
# Juga dibawahnya inisialisasi K yang akan digunakan pada algorita KNN
scores = []
k_neighbor = 5

# Dataset yang sudah displit tadi diloop dan dicari masing-masing akurasinya.
# Lalu masing-masing akurasinya disimpan ke dalam scores
for i in range(split):
	model = KNN_Model(train['x'][i], train['y'][i], k_neighbor)
	prediction = model.predict(test['x'][i])
	score = accuracy_score(test['y'][i], prediction)
	score = score * 100
	scores.append(score)
# Cari Mean (Rata-rata) score (akurasi) dari semua score yang sudah didapatkan
mean_score = sum(scores) / len(scores)

# Seluruh score yang didapatkan tadi juga dapat diplotkan dengan library matplotlib
# Agar perubahan akurasi yang didapatkan dapat dilihat dengan lebih baik dengan menggunakan grafik
plt.plot(range(split), scores)
plt.xlabel('Datasets Index')
plt.ylabel('Accuracy Score')
plt.text(0.0, 102.0, 'Mean Accuracy ' + str(round(mean_score,2)) + '%', fontsize=10)
plt.show()