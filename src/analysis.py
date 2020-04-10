import numpy as np
import matplotlib.pyplot as plt
import pickle

# Loading the svm classifier
with open('clf.pkl', 'rb') as f:
	clf = pickle.load(f)

# Generating feature importance plot
plt.bar(np.arange(4000), abs(clf.coef_[0]))
plt.title('Relative Importance of each flow word')
plt.show()

# Histogram of frame number vs important flow word
