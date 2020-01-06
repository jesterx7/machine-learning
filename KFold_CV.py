from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

def KNN_Model(x_train, y_train, k):
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(x_train, y_train)

	return knn

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

datasets = load_iris()
data_train = datasets.data
data_labels = datasets.target
split = 5
train, test = KFoldCV(data_train, data_labels, split)

scores = []
k_neighbor = 5
for i in range(split):
	model = KNN_Model(train['x'][i], train['y'][i], k_neighbor)
	prediction = model.predict(test['x'][i])
	score = accuracy_score(test['y'][i], prediction)
	score = score * 100
	scores.append(score)

plt.plot(range(split), scores)
plt.xlabel('Datasets Index')
plt.ylabel('Accuracy Score')
plt.show()