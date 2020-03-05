import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pickle

with open('centers.pkl', 'rb') as f:
	centers = pickle.load(f)	

NUM_OF_CLUSTERS = centers.shape[0]

# Extracting flows for the query video


query_video_vector = np.zeros(NUM_OF_CLUSTERS)

for i in range(np.shape(desc)[0]):
	distances = np.linalg.norm(centers - desc[i], axis = 1)
	belonging_cluster = np.argmin(distances)
	query_image_vector[belonging_cluster] += 1

# Passing through the svm
with open('clf.pkl', 'rb') as f:
	clf = pickle.load(f)

pred = clf.predict(query_image_vector)
if pred == '1':
	print('Forward video')
else:
	print('Backward video')