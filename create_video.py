import cv2
import numpy as np
import os
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pathIn', default='./ati_frames/')
parser.add_argument('--pathOut', default='output_video_single.avi')
parser.add_argument('--fps', default=10)

args = parser.parse_args()
pathIn = args.pathIn
pathOut = args.pathOut
fps = int(args.fps)

frame_array = []
files = [int(f[5:-4]) for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort()
for i in files:
    filename=pathIn + 'frame' + str(i) + '.jpg'
    print(filename)
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

for i in range(len(frame_array)):
	# writing to a image array
	out.write(frame_array[i])
out.release()