import numpy as np
from sklearn.svm import SVC
import pickle 
import os
from utils import NUM_OF_WORDS, TRAIN_PATH

def freq(curr_labels):
	hist = np.bincount(curr_labels[:, 0], minlength=NUM_OF_WORDS)
	return hist

# Generating the data
with open('dict.pkl', 'rb') as f:
	DICT = pickle.load(f)

with open('labels.pkl', 'rb') as f:
	labels = pickle.load(f)

X = []
Y = []

csum = 0
for i, folder in enumerate(sorted(os.listdir(TRAIN_PATH))):
    print("Folder ", folder, " in progress")
    
    # (A): the native direction of the video
    print("Flow A")
    if DICT[folder]['A'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['A']]
	    csum += DICT[folder]['A']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(1)
	    else:
	    	Y.append(-1)
	    
    # (B): this video mirrored in the left-right direction
    print("Flow B")
    if DICT[folder]['B'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['B']]
	    csum += DICT[folder]['B']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(1)
	    else:
	    	Y.append(-1)

    # (C): the original video time-flipped;
    print("Flow C")
    if DICT[folder]['C'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['C']]
	    csum += DICT[folder]['C']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(-1)
	    else:
	    	Y.append(1)

    # (D): the time-flipped left-right-mirrored version.
    print("Flow D")
    if DICT[folder]['D'] != 0:
	    curr_labels = labels[csum:csum+DICT[folder]['D']]
	    csum += DICT[folder]['D']
	    X.append(freq(curr_labels))
	    if folder[0] == 'F':
	    	Y.append(-1)
	    else:
	    	Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Normalizing features
X = X/(np.linalg.norm(X, axis=1)[:, None]+1e-10)
print('X.shape:', X.shape, 'Y.shape:', Y.shape)

# SVM
clf = SVC(gamma='scale', kernel='linear')
clf.fit(X, Y)

# Storing the classifier
with open('clf.pkl', 'wb') as f:
	pickle.dump(clf, f)

print('Training accuracy:', 100*np.sum(clf.predict(X)==Y)/X.shape[0])