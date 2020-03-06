import numpy as np
from sklearn.svm import SVC

# Parameters
DATA_PATH = '../ArrowDataAll/Train'
NUM_OF_WORDS = 4000

# Generating the data
with open('dict.pkl', 'rb') as f:
	DICT = pickle.load(f)

with open('labels.pkl', rb) as f:
	labels = pickle.load(f)

X = []
Y = []

csum = 0
for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
    print("Folder ", folder, " in progress")
    
    # (A): the native direction of the video
    print("Started Flow A")
    curr_labels = labels[csum:csum+DICT[folder]['A']]
    csum += DICT[folder]['A']
    X.append(np.bincount(curr_labels))
    if folder[0] == 'F':
    	Y.append(1)
    else:
    	Y.append(-1)
    
    # (B): this video mirrored in the left-right direction
    print("Flow B")
    curr_labels = labels[csum:csum+DICT[folder]['B']]
    csum += DICT[folder]['B']
    X.append(np.bincount(curr_labels))
    if folder[0] == 'F':
    	Y.append(1)
    else:
    	Y.append(-1)

    # (C): the original video time-flipped;
    print("Flow C")
    curr_labels = labels[csum:csum+DICT[folder]['C']]
    csum += DICT[folder]['C']
    X.append(np.bincount(curr_labels))
    if folder[0] == 'F':
    	Y.append(-1)
    else:
    	Y.append(1)

    # (D): the time-flipped left-right-mirrored version.
    print("Flow D")
    curr_labels = labels[csum:csum+DICT[folder]['D']]
    csum += DICT[folder]['D']
    X.append(np.bincount(curr_labels))
    if folder[0] == 'F':
    	Y.append(-1)
    else:
    	Y.append(1)

print('X.shape:', X.shape, 'Y.shape:', Y.shape)

# SVM
clf = SVC(gamma='auto')
clf.fit(X, y)

# Storing the classifier
with open('clf.pkl', 'wb') as f:
	pickle.dump(clf, f)