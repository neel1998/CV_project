import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pickle
import os
import multiprocessing as mp
import time

DATA_PATH = '../ArrowDataAll/Test'
NUM_OF_WORDS = 4000
THRESHOLD = 1000
STRIDE = 3
PATCH_IDX = []

def freq(curr_labels):
	global NUM_OF_WORDS

	try:
		hist = np.bincount(curr_labels, minlength=NUM_OF_WORDS)
	except:
		hist = np.bincount(np.zeros(0, dtype=np.int64), minlength=NUM_OF_WORDS)
	return hist

def generate_patches():
	patch_x, patch_y = [], []
	cnt = 0
	for y in range(0,552-4,3):
		for x in range(0,982-4,3):
			cnt += 1
			yy, xx = np.meshgrid(np.arange(x,x+4),np.arange(y,y+4))
			idx_x = xx.reshape(16)
			idx_y = yy.reshape(16)

			patch_x.append(idx_x)
			patch_y.append(idx_y)
	
	global PATCH_IDX
	PATCH_IDX = [tuple(patch_y), tuple(patch_x)]

def registration(img1, img2):
	sz = img1.shape
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	try:
		(cc, warp_matrix) = cv2.findTransformECC(img1,img2, warp_matrix)
		return cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	except:
		return img2

def _optical_flow(imgs):
	imgs, i = args
	img12 = registration(imgs[i], imgs[i-1])
	img23 = registration(imgs[i], imgs[i+1])

	flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
	
	flowX_patch = flow[:, :, 0][tuple(PATCH_IDX)]
	flowY_patch = flow[:, :, 1][tuple(PATCH_IDX)]
	flow_patch = np.concatenate((flowX_patch, flowY_patch), axis=1)
	fl = flow_patch[np.linalg.norm(flow_patch, axis=1) > THRESHOLD]
	fl = fl.flatten()

	if fl.shape[0] > 0:
		return fl
	else:
		return np.array([-1])

def optical_flow(imgs):
	""" Compute optical flow of t-1, t, t+1 images """
	p = mp.Pool(8)
	arr = [(imgs, i) for i in range(1, len(imgs) - 1)]
	flow = p.map(_optical_flow,arr)
	aa = np.concatenate(flow).tolist()
	res = np.array(aa)
	res = res[res!=-1].reshape(-1,32)
	p.close()
	p.join()

	return res

def getQueryVec(flow, centers):
	query_vector = np.zeros(centers.shape[0])
	# flow = optical_flow(imgs)

	dists = np.zeros((centers.shape[0], flow.shape[0]))
	dists = np.sqrt((flow**2).sum(axis=1)[:, np.newaxis] + (centers**2).sum(axis=1) - 2 * flow.dot(centers.T))
	labels = np.argmin(dists, axis=1)
	query_vector = freq(labels)
	return query_vector

def test():
	generate_patches()
	with open('clf.pkl', 'rb') as f:
		clf = pickle.load(f)
	
	with open('centers.pkl', 'rb') as f:
		centers = pickle.load(f)    


	correct = 0
	incorrect = 0
	for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
		print('Working on folder:', folder)
		imgs = []
		for j, files in enumerate(sorted(os.listdir(os.path.join(DATA_PATH,folder)))):
			print("Reading ", os.path.join(DATA_PATH,folder,files))
			img = cv2.imread(os.path.join(DATA_PATH,folder,files))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray,(552, 982))
			imgs.append(resized)
			
		imgs2 = [img[...,::-1,:] for img in imgs]

		flowA = optical_flow(imgs)
		flowB = optical_flow(imgs2)
		flowC = optical_flow(imgs[::-1])
		flowD = optical_flow(imgs2[::-1])

		flow = np.concatenate((flowA,flowB,flowC,flowD),axis=0)

		query_vector = getQueryVec(flow, centers).reshape(1, -1) 

		# Predicitng
		pred = clf.predict(query_vector)
		if pred == 1 and folder[0] == 'F' :
			correct += 1
		elif pred == -1 and folder[0] != 'F':
			correct += 1
		else:
			incorrect += 1

	acc = correct/(correct + incorrect)
	print("Testing Accuracy : " + str(acc))             

if __name__ == '__main__':
	test()