import argparse
import os
from skimage import io #library for python to help access pictures
import numpy as np #help do math in python
import matplotlib.pyplot as plt
import random
import imageio
import PIL
from PIL import Image
from skimage.util.shape import view_as_windows, view_as_blocks
import imutils
import os
import cv2
import glob
import shutil
import re
import pandas as pd
import pathlib
import time
import errno, os, stat, shutil

def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

Artifact = True

PIL.Image.MAX_IMAGE_PIXELS = 9331200000




image_path = glob.glob('Input/*.*')
image_path = image_path[0]


data = pd.read_csv('Prefrences/gdterminal_config', sep=",", header=None)
settings = np.array(data[1][0:])


yes = 'yes'
no = 'no'

if yes in settings[1]:
  sixnm = True
elif no in settings[1]:
  sixnm = False

if yes in settings[2]:
  twelvenm = True
elif no in settings[2]:
  twelvenm = False

if yes in settings[3]:
  eighteennm = True
elif no in settings[3]:
  eighteennm = False


lower_six = float(settings[4])
upper_six = float(settings[5])
lower_twelve = float(settings[6])
upper_twelve = float(settings[7])
lower_eighteen = float(settings[8])
upper_eighteen = float(settings[9])
thresh_sens = float(settings[10])





shutil.rmtree('Media/Output')
os.mkdir('Media/Output')

shutil.rmtree('pix2pix/datasets/Output_Appended/test')
os.mkdir('pix2pix/datasets/Output_Appended/test')

shutil.rmtree('pix2pix/results/golddigger/test_latest/images')
os.mkdir('pix2pix/results/golddigger/test_latest/images')

shutil.rmtree('Media/Output_ToStitch')
os.mkdir('Media/Output_ToStitch')


## look in INPUT folder, crop photo and save crop to OUTPUT folder
def load_data_make_jpeg(entry):
    img_size=(256,256, 3)
    img_new  = io.imread(entry)
    #img_new = (img_new/256).astype('uint8')
    shape = img_new.shape
    height = shape[0]//256
    height256 = height*256
    width = shape[1]//256
    width256 = width*256

    img_new = img_new[:height256,:width256,:3]


    img_new_w = view_as_blocks(img_new, img_size)
    imageio.imwrite('Output/' + 'CroppedProfile' + '.png', img_new)
    r = 0
    for i in range(img_new_w.shape[0]):
        for j in range(img_new_w.shape[1]):
            A = np.zeros((img_size[0], img_size[1], 3))
            A[:,:,:] = img_new_w[i,j,:,:]
            imageio.imwrite('Media/Output/'+ str(r) + '.png', A)
            r += 1
    return width, height


## Cut up in order, append white images
width, height = load_data_make_jpeg(image_path)

def combine_white(folderA):
    
    for filepath in os.listdir(folderA):
        imA = io.imread(folderA + filepath)
        newimage = np.concatenate((imA,white), axis=1)
        imageio.imwrite('pix2pix/datasets/Output_Appended/test/' + filepath, newimage)

white = io.imread('white/white.png')

combine_white('Media/Output/')

## Save that dataset to PIX2PIX/datasets/___

## Run PIX2PIX network
os.system('python3 pix2pix/test.py --dataroot pix2pix/datasets/Output_Appended --name golddigger --model pix2pix --direction AtoB --num_test 1000000 --checkpoints_dir pix2pix/checkpoints/ --results_dir pix2pix/results/')
## Take only the Fake_B photos and stich together
list = glob.glob('pix2pix/results/golddigger/test_latest/images/*_fake_B.png')
## Save to OUTPUT folder
for entry in list:
    split_name = entry.split('/')
    

    dirA = 'pix2pix/results/golddigger/test_latest/images/'
    pathA = os.path.join(dirA,split_name[5])
    dirB = 'Media/Output_ToStitch/'
    pathB = os.path.join(dirB,split_name[5])
    shutil.move(pathA, pathB)

## STICH TOGETHER


widthdiv256 = width
heighttimeswidth =  width * height

folderstart = 'Media/Output_ToStitch/'
def stitch_row(n):
    image1 = imageio.imread(folderstart+master[n])
    if (Artifact):
        image1[0:35, 220:256, :] = 0
    file1 = np.array(image1)

    image2 = imageio.imread(folderstart+master[n+1])
    if (Artifact):
        image2[0:35, 220:256, :] = 0
    file2 = np.array(image2)

    full_row = np.concatenate((file1, file2), axis=1)
    for i in range(n + 2, n + widthdiv256):
        image3 = imageio.imread(folderstart + master[i])
        if (Artifact):
            image3[0:35,220:256, :] = 0
        file_next = np.array(image3)
        full_row = np.concatenate((full_row, file_next), axis = 1)
    return full_row

files = os.listdir(folderstart)
list = []
for file in files:
    split_name = re.split('\D', file)

    list.append(split_name[0])

list.sort(key = float)
master = []
for file in list:
    name = file + '_fake_B.png'
    master.append(name)


picture = stitch_row(0)
for n in range(widthdiv256,heighttimeswidth,widthdiv256):
    next_row = stitch_row(n)
    picture = np.concatenate((picture,next_row), axis=0)

imageio.imwrite('Media/Output_Final/OutputStitched.png', picture)
## Count All Green Dots

img = cv2.imread('Media/Output_Final/OutputStitched.png')
img_original = cv2.imread('Output/CroppedProfile.png')
img_original = np.uint8(img_original)

h, w = img_original.shape[:2]
flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)

lower_green = np.array([0,245,0])
upper_green = np.array([40,255,40])


mask = cv2.inRange(img, lower_green, upper_green)
kernel = np.ones((5,5), np.uint8)
e = cv2.erode(mask, kernel, iterations=1)
d = cv2.dilate(e, kernel, iterations = 1)


cnts = cv2.findContours(d,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
seedlistx  = []
seedlisty = []
for c in cnts:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        M["m00"] = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    if (cX != 0 or cY != 0):
        if img_original[cY,cX,0] < 120:
            seedlistx.append(cX)
            seedlisty.append(cY)


listlen = len(seedlistx)
floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)
for i in range(listlen):
    num, im, mask, rect = cv2.floodFill(img_original, flood_mask, (seedlistx[i], seedlisty[i]),1, (thresh_sens,)*3, (thresh_sens,)*3, floodflags)


classes = []
classes.append(sixnm)
classes.append(twelvenm)
classes.append(eighteennm)
num_classes = sum(classes)
if num_classes == 1:
    if sixnm:
        single_lower_bound = lower_six
        single_upper_bound = upper_six
    if twelvenm:
        single_lower_bound = lower_twelve
        single_upper_bound = upper_twelve
    if eighteennm:
        single_lower_bound = lower_eighteen
        single_upper_bound = upper_eighteen


flood_mask = flood_mask[:h,:w]
cnts = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if sixnm:
    results6 = pd.DataFrame(columns = ['X','Y'])
if twelvenm:
    results12 = pd.DataFrame(columns = ['X','Y'])
if eighteennm:
    results18 = pd.DataFrame(columns = ['X','Y'])

for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        M["m00"] = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    if num_classes == 1:
        if cv2.contourArea(c) < single_upper_bound and cv2.contourArea(c) > single_lower_bound:
            if sixnm:
                results6 = results6.append({'X': cX, 'Y': cY}, ignore_index=True)
            elif twelvenm:
                results12 = results12.append({'X': cX, 'Y': cY}, ignore_index=True)
            elif eighteennm:
                results18 = results18.append({'X': cX, 'Y': cY}, ignore_index=True)
    if num_classes == 2:
        if cv2.contourArea(c) < upper_six and cv2.contourArea(c) >= lower_six:
            if sixnm:
                results6 = results6.append({'X': cX, 'Y': cY}, ignore_index=True)
            elif twelvenm:
                results12 = results12.append({'X': cX, 'Y': cY}, ignore_index=True)
        if cv2.contourArea(c) < upper_twelve and cv2.contourArea(c) >= lower_twelve:
            if twelvenm:
                results12 = results12.append({'X': cX, 'Y': cY}, ignore_index=True)
            elif eighteennm:
                results18 = results18.append({'X': cX, 'Y': cY}, ignore_index=True)
    if num_classes == 3:
        if cv2.contourArea(c) < upper_six and cv2.contourArea(c) >= lower_six:
            results6 = results6.append({'X': cX, 'Y': cY}, ignore_index=True)
        if cv2.contourArea(c) < upper_twelve and cv2.contourArea(c) >= lower_twelve:
            results12 = results12.append({'X': cX, 'Y': cY}, ignore_index=True)
        if cv2.contourArea(c) < upper_eighteen and cv2.contourArea(c) >= lower_eighteen:
            results12 = results18.append({'X': cX, 'Y': cY}, ignore_index=True)





if sixnm:
    export_csv = results6.to_csv(r'Output/Results6nm.csv', index = None, header= True)

if twelvenm:
    export_csv = results12.to_csv(r'Output/Results12nm.csv', index = None, header= True)

if eighteennm:
    export_csv = results18.to_csv(r'Output/Results18nm.csv', index = None, header= True)


shutil.rmtree('Input')
os.mkdir('Input')



