from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

def KNN_Model(x_train, y_train):
	knn = KNeighborsClassifier(n_neighbors = 5)
	knn.fit(x_train, y_train)

	return knn

def KFoldCV(data_x, data_y):
	kf = KFold(n_splits = 5)
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

train, test = KFoldCV(data_train, data_labels)

scores = []
for i in range(5):
	model = KNN_Model(train['x'][i], train['y'][i])
	prediction = model.predict(test['x'][i])
	score = accuracy_score(test['y'][i], prediction)
	score = score * 100
	scores.append(score)

plt.plot(range(5), scores)
plt.xlabel('Datasets Index')
plt.ylabel('Accuracy Score')
plt.show()