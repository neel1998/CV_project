import numpy as np
import cv2
from os import path
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

DATA_PATH = '../ArrowDataAll'
TRAIN_PATH = path.join(DATA_PATH, 'Train')
TEST_PATH = path.join(DATA_PATH, 'Test')
STRIDE = 3
NUM_OF_WORDS = 4000
h, w = 552, 982

class Utils():

	def __init__(self):
		self.Patch_idx = []
		self.imgs = None
		self.THRESHOLD = 1300

	def generate_patches(self):
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
		
		self.Patch_idx = tuple([tuple(patch_x), tuple(patch_y)])

	def registration(self, img1, img2):
		sz = img1.shape
		warp_matrix = np.eye(2, 3, dtype=np.float32)
		try:
			(cc, warp_matrix) = cv2.findTransformECC(img1,img2, warp_matrix)
			return cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		except:
			return img2


	def _optical_flow(self, i):
		img12 = self.registration(self.imgs[i], self.imgs[i-1])
		img23 = self.registration(self.imgs[i], self.imgs[i+1])

		flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
		
		flowX_patch = flow[:, :, 0][self.Patch_idx]
		flowY_patch = flow[:, :, 1][self.Patch_idx]
		flow_patch = np.concatenate((flowX_patch, flowY_patch), axis=1)
		fl = flow_patch[np.linalg.norm(flow_patch, axis=1) > self.THRESHOLD]
		fl = fl.flatten()

		if fl.shape[0] > 0:
			return fl
		else:
			return np.array([-1])

	def optical_flow(self, imgs):
		""" Compute optical flow of t-1, t, t+1 images """
		self.imgs = imgs
		flow = []
		for i in range(1, len(self.imgs) - 1):
			flow.append(self._optical_flow(i))

		aa = np.concatenate(flow).tolist()
		res = np.array(aa)
		res = res[res!=-1].reshape(-1,32)
		return res