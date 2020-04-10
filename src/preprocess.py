import os
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from utils import TRAIN_PATH, h, w, Utils

if __name__ == '__main__':

	cv2.setUseOptimized(True)
	cv2.setNumThreads(4)
	
	DICT = {}
	util = Utils()
	util.generate_patches()

	for i, folder in enumerate(sorted(os.listdir(TRAIN_PATH))):
		imgs = []
		for j, files in enumerate(sorted(os.listdir(os.path.join(TRAIN_PATH,folder)))):
			# print("Reading ", os.path.join(TRAIN_PATH,folder,files))
			img = cv2.imread(os.path.join(TRAIN_PATH,folder,files))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray,(982, 552))
			imgs.append(resized)
		
		print("Folder ", folder, " in progress")		
		DICT[folder] = {}

		imgs2 = [np.fliplr(img) for img in imgs]

		# (A): the native direction of the video
		t = time.time()
		print("Started Flow A")
		flowA = util.optical_flow(imgs)
		DICT[folder]['A'] = flowA.shape[0] 

		# (B): this video mirrored in the left-right direction
		print("Started Flow B")
		flowB = util.optical_flow(imgs2)
		DICT[folder]['B'] = flowB.shape[0]

		# (C): the original video time-flipped;
		print("Started Flow C")
		flowC = util.optical_flow(imgs[::-1])
		DICT[folder]['C'] = flowC.shape[0]

		# (D): the time-flipped left-right-mirrored version.
		print("Started Flow D")
		flowD = util.optical_flow(imgs2[::-1])
		DICT[folder]['D'] = flowD.shape[0]

		# Adding to the main matrix
		flow = np.concatenate((flowA,flowB,flowC,flowD),axis=0)
		if i == 0:
		    flows = flow
		else:
		    flows = np.concatenate((flows, flow),axis=0)

		print(DICT)
		print('Shape of flows:', flows.shape)
		
	pickle_out = open("dict.pkl","wb")
	pickle.dump(DICT, pickle_out)
	pickle_out.close()

	pickle_out = open("patch_flow.pkl","wb")
	pickle.dump(flows, pickle_out)
	pickle_out.close()