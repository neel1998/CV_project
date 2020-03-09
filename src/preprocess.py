import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import multiprocessing as mp

DATA_PATH = '../ArrowDataAll/Train'
THRESHOLD = 1200
STRIDE = 3
PATCH_IDX = []

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
	img12 = registration(imgs[1], imgs[0])
	img23 = registration(imgs[1], imgs[2])

	flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
	return flow

def optical_flow(imgs):
	""" Compute optical flow of t-1, t, t+1 images """
	flows = []
	for i, _ in enumerate(imgs):
		if i > 0 and i < len(imgs) - 1:
			flow = _optical_flow([imgs[i-1],imgs[i],imgs[i+1]])
			flowX_patch = flow[:, :, 0][tuple(PATCH_IDX)]
			flowY_patch = flow[:, :, 1][tuple(PATCH_IDX)]
			flow_patch = np.concatenate((flowX_patch, flowY_patch), axis=1)
			valid_flow = flow_patch[np.linalg.norm(flow_patch, axis=1) > THRESHOLD]
			if i == 1:
				valid_flows = valid_flow
			else:
				valid_flows = np.concatenate((valid_flows,valid_flow), axis = 0)
	return valid_flows


generate_patches()
if __name__ == '__main__':

	DICT = {}
	pool = mp.Pool(4)

	for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
		imgs = []
		for j, files in enumerate(sorted(os.listdir(os.path.join(DATA_PATH,folder)))):
			# print("Reading ", os.path.join(DATA_PATH,folder,files))
			img = cv2.imread(os.path.join(DATA_PATH,folder,files))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray,(552, 982))
			imgs.append(resized)
				
		print("Folder ", folder, " in progress")
		DICT[folder] = {}

		imgs2 = [img[...,::-1,:] for img in imgs]
		t = time.time()
		print("Started Flow All")
		res = pool.map(optical_flow, [imgs,imgs2,imgs[::-1],imgs2[::-1]])

		# (A): the native direction of the video
		# print("Flow A")
		flowA = res[0]
		DICT[folder]['A'] = flowA.shape[0]
		print(flowA.shape)

		# (B): this video mirrored in the left-right direction
		# print("Flow B")
		flowB = res[1]
		DICT[folder]['B'] = flowB.shape[0]
		print(flowB.shape)
		
		# (C): the original video time-flipped;
		# print("Flow C")
		flowC = res[2]
		DICT[folder]['C'] = flowC.shape[0]
		print(flowC.shape)
		
		# (D): the time-flipped left-right-mirrored version.
		# print("Flow D")
		flowD = res[3]
		DICT[folder]['D'] = flowD.shape[0]
		print(flowD.shape)
		print("Complete", folder," Time elapsed ", time.time() - t, " seconds ")

		# Adding to the main matrix
		flow = np.concatenate((flowA,flowB,flowC,flowD),axis=0)
		if i == 0:
			flows = flow
		else:
			flows = np.concatenate((flows, flow),axis=0)

		print('Shape of flows:', flows.shape)
		print("time: ",time.time() - t)
	
	pickle_out = open("dict.pkl","wb")
	pickle.dump(DICT, pickle_out)
	pickle_out.close()

	pickle_out = open("patch_flow.pkl","wb")
	pickle.dump(flows, pickle_out)
	pickle_out.close()
