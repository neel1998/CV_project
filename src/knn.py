import numpy as np
import pickle 
import os
from sklearn.svm import SVC
from utils import NUM_OF_WORDS, TRAIN_PATH
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

print("import finished")
def freq(curr_labels):
	hist = np.bincount(curr_labels[:, 0], minlength=NUM_OF_WORDS)
	return hist

# Generating the data
with open('dict.pkl', 'rb') as f:
	DICT = pickle.load(f)

with open('labels.pkl', 'rb') as f:
	labels = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    X_test = pickle.load(f)[:,0,:]

with open('test_labels.pkl', 'rb') as f:
    Y_test = pickle.load(f)

print("X_test.shape:", X_test.shape, "Y_test.shape:", Y_test.shape)    
X = []
Y = []

csum = 0
for i, folder in enumerate(sorted(os.listdir(TRAIN_PATH))):
    
    # (A): the native direction of the video
    if DICT[folder]['A'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['A']]
	    csum += DICT[folder]['A']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(1)
	    else:
	    	Y.append(0)
	    
    # (B): this video mirrored in the left-right direction
    if DICT[folder]['B'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['B']]
	    csum += DICT[folder]['B']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(1)
	    else:
	    	Y.append(0)

    # (C): the original video time-flipped;
    if DICT[folder]['C'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['C']]
	    csum += DICT[folder]['C']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(0)
	    else:
	    	Y.append(1)

    # (D): the time-flipped left-right-mirrored version.
    if DICT[folder]['D'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['D']]
	    csum += DICT[folder]['D']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(0)
	    else:
	    	Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Normalizing features
X = X/(np.linalg.norm(X, axis=1)[:, None]+1e-10)
print('X.shape:', X.shape, 'Y.shape:', Y.shape)

# pca = PCA(n_components = 450)
# X = pca.fit_transform(X)

neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(X, Y)

print("KNN results:")
correct = 0
for i in range(X.shape[0]):
    d = X[i]
    if Y[i] == neigh.predict([d]):
        correct += 1

print("Training acc : " + str(correct/Y.shape[0]))


with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

correct = 0
for i in range(X_test.shape[0]):
    d = X_test[i]
    # pred = 
    if Y_test[i] == neigh.predict([d]):
        correct += 1

print("Testing acc : " + str(correct/Y_test.shape[0]))