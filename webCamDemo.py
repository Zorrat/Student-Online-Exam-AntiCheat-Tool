import os
import argparse
import cv2
import time
import colorsys
import numpy as np
import moviepy
from moviepy.editor import *


NETWORK_W = 608
NETWORK_H = 608

######## CLI Argparser ########
parser = argparse.ArgumentParser(description='Command options for Student-AntiCheat Tool')


parser.add_argument('--n','--name', type=str,
                    help='Required: Name of Student to check for exam.  Please add students image in faces Folder with appropriate name. (Specify in double " " quotes) ',
                    required=True,metavar='')   
parser.add_argument('--m','--phone',action='store_true',
                    help='Specify this flag for Cell Phone cheating Detection. WARNING Uses YoloV4 algorithm and impacts performance Heavily' )
parser.add_argument('--v','--verbose',action = 'store_true',
                    help='Specify this flag to get Verbose output.')
args = parser.parse_args()

if len(os.listdir('Faces')) == 0:
    raise Exception('No Face files detected in Faces folder. Please populate the directory.')


import face_recognition as faceRec
#### Importing Heavy utils later after argparse for performance #########
import antiCheatUtils as aUtils


labels = aUtils.read_labels("models/yolo/coco_classes.txt")
#class_threshold = 0.6
colors = aUtils.generate_colors(labels)
 
########### GET Face Data from Faces Folder for Face Recognition ############
faceDir = 'Faces'
faceImages = []
faceNames = []
faceEncodingsKnown = []
fileNames = os.listdir(faceDir)
for fn in fileNames:
    if fn[-3:]  in ['png','jpg']:
        faceIm = faceRec.load_image_file(faceDir + '/' +fn)
        faceEnc = faceRec.face_encodings(faceIm)[0]
        faceImages.append(faceIm)
        faceEncodingsKnown.append(faceEnc)
        faceName = fn[:-4]
        faceNames.append(faceName)
    
print('Faces Found in Faces Directory')
print(*faceNames)

cap = cv2.VideoCapture(0)
nameToCheck = args.n
absentFramesTotal,phoneFramesTotal = 0,0 
fc = 0 

print('Press (q) To abort')
tic = time.time()
while(cap.isOpened()):
    ret, img = cap.read()
    #if ret==True:
    fc+=1
    img,absentFramesTotal = aUtils.faceRecInference(faceEncodingsKnown,faceNames,img,absentFramesTotal,nameToCheck)
    if args.m:
        img, width,height = aUtils.load_image_pixels(img, (608,608))
        img,phoneFramesTotal = aUtils.Inference(img,width,height,colors,labels,phoneFramesTotal)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('feed',img)
    if args.v:
        print(f'Processing Frame number: {fc}')
        if args.m:
            print(f'Number of frames where phone was detected: {phoneFramesTotal}')
        print(f'Number of Frames where Student was missing: {absentFramesTotal}')
        print('-------------------------------------------------------------')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
toc = time.time()

print('Total Run Time', round((toc-tic)/60,2),'min')


cv2.destroyAllWindows()
cap.release()

print('Total frames in capture: ',fc)
print('Total number of Frames where the student was missing: ',absentFramesTotal)
print('Percent of time where the student was present',round(((fc-absentFramesTotal)/fc)*100,2))
if args.m:
    print('Total number of Phone Frames : ',phoneFramesTotal)
    print('Percent of time when cellphone use was detected: ', round((phoneFramesTotal/fc)*100,2))