import cv2
import numpy as np
import glob
import pickle

# As used in paper
NUM_OF_CLUSTERS = 4000

# Loading all the flows of the images
with open('patch_flow.pkl', 'rb') as f:
	flows = pickle.load(f)

# define criteria and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
ret, labels, centers = cv2.kmeans(flows, NUM_OF_CLUSTERS, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)

# Pickling the centers
with open('centers.pkl', 'wb') as f:
	pickle.dump(centers, f)

# Pickling the labels
with open('labels.pkl', 'wb') as f:
	pickle.dump(labels, f)
	
print('xxxxxxxxxxxxxxx Clustering completed xxxxxxxxxxxxxxxxxxxxxxx')