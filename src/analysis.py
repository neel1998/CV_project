import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2

def generate_patches(h, w):
	patch_x, patch_y = [], []
	cnt = 0
	for y in range(0,h-4,3):
		for x in range(0,w-4,3):
			cnt += 1
			yy, xx = np.meshgrid(np.arange(x,x+4),np.arange(y,y+4))
			idx_x = xx.reshape(16)
			idx_y = yy.reshape(16)

			patch_x.append(idx_x)
			patch_y.append(idx_y)
	
	return tuple([tuple(patch_x), tuple(patch_y)])

def registration(img1, img2):
	sz = img1.shape
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	try:
		(cc, warp_matrix) = cv2.findTransformECC(img1,img2, warp_matrix)
		return cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	except:
		return img2


def _optical_flow(imgs, i, THRESHOLD, Patch_idx):
	img12 = registration(imgs[i], imgs[i-1])
	img23 = registration(imgs[i], imgs[i+1])

	flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
	
	flowX_patch = flow[:, :, 0][Patch_idx]
	flowY_patch = flow[:, :, 1][Patch_idx]

	flow_patch = np.concatenate((flowX_patch, flowY_patch), axis=1)
	fl = flow_patch[np.linalg.norm(flow_patch, axis=1) > THRESHOLD]
	fl = fl.flatten()

	if fl.shape[0] > 0:
		return fl
	else:
		return np.array([-1])

def freq(curr_labels, NUM_OF_WORDS):
	hist = np.bincount(curr_labels, minlength=NUM_OF_WORDS)
	return hist

def getImpFeatureCount(flow, centers, imp_feature, NUM_OF_WORDS):
	dists = np.zeros((centers.shape[0], flow.shape[0]))
	dists = np.sqrt((flow**2).sum(axis=1)[:, np.newaxis] + (centers**2).sum(axis=1) - 2 * flow.dot(centers.T))
	
	labels = np.argmin(dists, axis=1)
	featureCount = freq(labels, NUM_OF_WORDS)
	
	return np.sum(featureCount[imp_features])

# Parameters
h, w = 552, 982
NUM_OF_WORDS = 4000
THRESHOLD = 1300
Patch_idx = generate_patches(h, w)

# Loading the svm classifier
with open('clf.pkl', 'rb') as f:
	clf = pickle.load(f)

# Generating feature importance plot
plt.bar(np.arange(NUM_OF_WORDS), abs(clf.coef_[0]))
plt.title('Relative Importance of each flow word')
plt.show()

# Histogram of frame number vs important flow word
# Results to be displayed on this video
folder = '../ArrowDataAll/Test/B1tREHGodIEZk'

# Finding the most important feature
imp_features = np.argsort(np.abs(clf.coef_[0]))[::-1][:400]

with open('centers.pkl', 'rb') as f:
	centers = pickle.load(f)

imgs = []
for j, files in enumerate(sorted(os.listdir(folder))):
	img = cv2.imread(os.path.join(folder,files))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray,(w, h))
	imgs.append(resized)

count = []
for i in range(1, len(imgs)-1):
	print('Frame number:', i)
	frameFlow = _optical_flow(imgs, i, THRESHOLD, Patch_idx)
	if frameFlow.shape[0] == 1:
		count.append(0)
		continue

	frameFlow = frameFlow.reshape(-1, 32)
	print(frameFlow.shape)
	countImpFeature = getImpFeatureCount(frameFlow, centers, imp_features, NUM_OF_WORDS)
	print(countImpFeature)

	count.append(countImpFeature)

# Plotting bar graph
print(count)
print(len(imgs)-2, len(count))
plt.bar(np.arange(len(imgs)-2), count)
plt.title('Presence of indicative words')
plt.show()