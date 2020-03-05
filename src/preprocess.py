import os
import cv2
import numpy as np
import pickle

DATA_PATH = '../ArrowDataAll/'
THRESHOLD = 1000


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
    """ Compute optical flow of t, t-1, t+1 images """

    imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in imgs]

    flows = []
    for i, _ in enumerate(imgs):
        if i > 0 and i < len(imgs) - 1:
            flow = _optical_flow([imgs[i-1],imgs[i],imgs[i+1]])
            # print(flow.shape)

            for y in range(0,flow.shape[0]-4,4):
                for x in range(0,flow.shape[1]-4,4):
                    
                    patch = flow[y:y+4,x:x+4]
                    # print(np.linalg.norm(patch))
                    if np.linalg.norm(patch) > THRESHOLD:
                        flows.append(patch.flatten())
            
            # print(len(flows))

    flows = np.array(flows)
    return flows

DICT = {}

for i, folder in enumerate(sorted(os.listdir(DATA_PATH))):
    imgs = []
    for files in sorted(os.listdir(os.path.join(DATA_PATH,folder))):
        # print("Reading ", os.path.join(DATA_PATH,folder,files))
        imgs.append(cv2.imread(os.path.join(DATA_PATH,folder,files)))

    flow = optical_flow(imgs)
    if i == 0:
        flows = flow
    else:
        flows = np.vstack((flows, flow))

    DICT[folder] = flow.shape[0]
    
pickle_out = open("dict.pickle","wb")
pickle.dump(DICT, pickle_out)
pickle_out.close()

pickle_out = open("patch_flow.pickle","wb")
pickle.dump(flows, pickle_out)
pickle_out.close()