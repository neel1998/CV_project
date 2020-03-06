import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pickle


DATA_PATH = '../ArrowDataAll/Test'

THRESHOLD = 1000
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
    flows = []
    for i, _ in enumerate(imgs):
        if i > 0 and i < len(imgs) - 1:
        	flow = _optical_flow([imgs[i-1],imgs[i],imgs[i+1]])
        	flowX_patch = flow[:, :, 0][tuple(PATCH_IDX)]
        	flowY_patch = flow[:, :, 1][tuple(PATCH_IDX)]
        	flow_patch = np.concatenate((flowX_patch, flowY_patch), axis=1)
        	valid_flow = flow_patch[np.linalg.norm(flow_patch, axis=1) > THRESHOLD]

    return valid_flow


def getQueryVec(imgs, centers):
	query_vector = np.zeros(centers.shape[0])
	flow = optical_flow(imgs)
	for i in range(flow.shape[0]):
		distances = np.linalg.norm(centers - flow[i], axis = 1)
		belonging_cluster = np.argmin(distances)
		query_vector[belonging_cluster] += 1
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
	    imgs = []
	    for j, files in enumerate(sorted(os.listdir(os.path.join(DATA_PATH,folder)))):
	        # print("Reading ", os.path.join(DATA_PATH,folder,files))
	        img = cv2.imread(os.path.join(DATA_PATH,folder,files))
	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        resized = cv2.resize(gray,(552, 982))
	        imgs.append(resized)
		
	    query_ector = getQueryVec(imgs, centers)    

		pred = clf.predict(query_vector)
		if pred == '1' and folder[0] == 'F' :
			correct += 1
		elif pred == '-1' and folder[0] != 'F':
			correct += 1
		else:
			incorrect += 1	
	acc = correct/(correct + incorrect)
	print("Testing Accuracy : " + str(acc))	            

if __name__ == '__main__':
	test()