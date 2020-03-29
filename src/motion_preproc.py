import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
from preprocess import registration

# Function definitions
def my_unit_circle(r):
	d = 20
	rx, ry = d/2, d/2
	x, y = np.indices((d, d))
	return (np.hypot(rx - x, ry - y) <= r).astype(int)

# Parameters
DATA_PATH = '../ArrowDataAll/Train'
radii = [5, 6.6, 8.7]
threshold = [0.1, 0.01, 0.056]

# Loading the videos
for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
	print('Processing Video Number:', i+1)
	images = []
	for j, files in enumerate(os.listdir(os.path.join(DATA_PATH, folder))):
		img = cv2.imread(os.path.join(DATA_PATH,folder,files))
		images.append(img)

	print('Number of images:', len(images))

	for j in range(1, len(images)):
		# Image at t+1 and image at t
		I1 = images[j].astype(np.float32)
		I0 = images[j-1].astype(np.float32)

		# Registering the images
		W1 = registration(I0, I1)

		diff = np.abs(W1-I0)
		diff = np.sum(diff, axis=2)
		diff = cv2.resize(diff, (400, 400))
		for k in range(1):
			kernel = my_unit_circle(radii[k])
			diff = cv2.filter2D(diff, -1, kernel)
			diff[diff<=threshold[k]] = 0
			plt.gray()
			plt.imshow(diff)
			plt.show()
		break