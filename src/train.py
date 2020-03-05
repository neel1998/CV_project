import numpy as np
from sklearn.svm import SVC

# Generating the data
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

# SVM
clf = SVC(gamma='auto')
clf.fit(X, y)

# Storing the classifier
with open('clf.pkl', 'wb') as f:
	pickle.dump(clf, f)