import cv2
import os
import numpy as np
import subprocess
from utils import DATA_PATH, Utils

def makeVideo(pathIn,pathOut,fps,cond=False):
    
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    files.sort(key = lambda x: int(x[5:-5]))
    if cond:
    	files = files[1:-1]

    for i in range(len(files)):
        filename=os.path.join(pathIn,files[i])
        img = cv2.imread(filename)
        img = cv2.resize(img,(982, 552))
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
        
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame in frame_array:
        out.write(frame)
    out.release()

# util = Utils()
# util.generate_patches()

folder = 'Train/F_05gGCvIopwE'
name = folder.split('/')[1]
fps = 1

try:
	os.mkdir('../flow')
	os.mkdir(os.path.join('../flow',name))
except:
	pass

imgs = []
for files in sorted(os.listdir(os.path.join(DATA_PATH,folder))):

	img = cv2.imread(os.path.join(DATA_PATH,folder,files))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray,(982, 552))
	imgs.append(resized)

_ = util.optical_flow(imgs,True,name)

output_vid = os.path.join('../flow',name+'.mp4')
output_flowvid = os.path.join('../flow',name+'_flow.mp4')
combined_vid = os.path.join('../flow',name+'_final.mp4')

makeVideo(os.path.join(DATA_PATH,folder), output_vid, fps, True)
makeVideo(os.path.join('../flow',name), output_flowvid, fps)

command = 'ffmpeg -i {} -i {} -filter_complex hstack {}'.format(output_vid, output_flowvid, combined_vid)
subprocess.call(command, shell=True)