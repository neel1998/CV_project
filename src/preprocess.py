import os
import cv2
import numpy as np
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

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
	PATCH_IDX = [tuple(patch_x), tuple(patch_y)]

def registration(img1, img2):
	sz = img1.shape
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	try:
		(cc, warp_matrix) = cv2.findTransformECC(img1,img2, warp_matrix)
		return cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	except:
		return img2

def _optical_flow(args):
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


generate_patches()
if __name__ == '__main__':

	DICT = {}

	for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
		imgs = []
		for j, files in enumerate(sorted(os.listdir(os.path.join(DATA_PATH,folder)))):
			# print("Reading ", os.path.join(DATA_PATH,folder,files))
			img = cv2.imread(os.path.join(DATA_PATH,folder,files))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray,(982, 552))
			imgs.append(resized)
		
		print("Folder ", folder, " in progress")		
		DICT[folder] = {}

		imgs2 = [np.fliplr(img) for img in imgs]
		t = time.time()

		# cv2.imwrite('A.jpeg', imgs[0])
		# cv2.imwrite('B.jpeg', imgs2[0])
		# cv2.imwrite('C.jpeg', imgs[::-1][0])
		# cv2.imwrite('D.jpeg', imgs2[::-1][0])

		ptch = imgs[0][tuple(PATCH_IDX)]
		print(ptch.shape)
		cnt = 0
		tt = np.zeros(imgs[0].shape)
		for y in range(0,552-4,3):
			for x in range(0,982-4,3):
				tt[y:y+4,x:x+4] = ptch[cnt].reshape(4,4)
				cnt += 1
		# for i in range(10):
		# 	for j in range(10):
		# 		plt.subplot(10,10,10*i+j+1)
		# 		plt.imshow(ptch[10*i+j].reshape(4,4))
		plt.imshow(tt)
		plt.show()
		break
		# (A): the native direction of the video
		# print("Flow A")
		print("Started Flow A")
		flowA = optical_flow(imgs)
		DICT[folder]['A'] = flowA.shape[0] 
		
		# (B): this video mirrored in the left-right direction
		# print("Flow B")
		print("Started Flow B")
		flowB = optical_flow(imgs2)
		DICT[folder]['B'] = flowB.shape[0]

		# (C): the original video time-flipped;
		# print("Flow C")
		print("Started Flow C")
		flowC = optical_flow(imgs[::-1])
		DICT[folder]['C'] = flowC.shape[0]

		# (D): the time-flipped left-right-mirrored version.
		# print("Flow D")
		print("Started Flow D")
		flowD = optical_flow(imgs2[::-1])
		DICT[folder]['D'] = flowD.shape[0]

		print("Complete", folder," Time elapsed ", time.time() - t, " seconds ")

		# Adding to the main matrix
		flow = np.concatenate((flowA,flowB,flowC,flowD),axis=0)
		if i == 0:
		    flows = flow
		else:
		    flows = np.concatenate((flows, flow),axis=0)

		print('Shape of flows:', flows.shape)
		
	# pickle_out = open("dict.pkl","wb")
	# pickle.dump(DICT, pickle_out)
	# pickle_out.close()

	# pickle_out = open("patch_flow.pkl","wb")
	# pickle.dump(flows, pickle_out)
	# pickle_out.close()
