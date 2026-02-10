#-----------------FOLDER INFORMATION----------------
import os
from keras.preprocessing import image
import argparse
import cv2
# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import data, io
from skimage.util import img_as_ubyte
from PIL import Image

import colorama
from colorama import Fore, Back
import colorsys

def key_func(x):
  return os.path.split(x)[-1]

dir = "/content/gdrive/MyDrive/yolov7/crop" #all imgs

initial_count = 0
Cluster_1=0
Cluster_2=0


folderPath = os.path.join(dir)
for path in sorted(os.listdir(dir)):
  img =io.imread(os.path.join(dir, path))
  ori =io.imread(os.path.join(dir, path))
  img= cv2.resize(img, (64, 64))
  img = np.asarray(img)
  #plt.imshow(img),plt.title("Predicted Result"), plt.axis('off')
  img = np.expand_dims(img, axis=0)

  from keras.models import load_model
  saved_model = load_model(r"/content/gdrive/MyDrive/yolov7/model_test/BEST_MODEL-VGG16.h5")

  output = saved_model.predict(img)

  if output[0][0] > output[0][1]:
    #print("GoodStaining")
    Cluster_1 += 1
  #elif output[0][1] > output[0][2]:
    #Less += 1
    #print('LessStaining')
  else:
    Cluster_2 += 1
    #print('OverStaining')

  initial_count += 1

print('\033[1m' +Fore.BLACK+"Result: ")
print('\033[1m' +Fore.BLACK+"Path ID           :",os.path.basename(dir))
print('\033[1m' +Fore.BLACK+"Number of Datasets:",initial_count)

print(" ")
print('\033[1m' +Fore.BLACK+ "Second Detection: ")
print("Cluster (1) Red AFB:",Cluster_1, 'image(s)')
#print("Less Staining:",Less, 'image(s)')
print("Cluster (2) Artefact:",Cluster_2, 'image(s)')


'''
  if os.path.isfile(os.path.join(dir, path)):
    plt.figure(num=None, figsize=(16,16))
    plt.subplot(1,2,1), plt.title(os.path.basename(path)), plt.imshow(ori), plt.axis('off')
    #plt.subplot(1,2,2), plt.title("Predicted Result"), plt.imshow(img), plt.axis('off')
    # show the output image
'''
print(" ")
print("---------------------------------")

if output[0][0] > output[0][1]: #TB Scanty
  #secondstage...........********************************************************************************
  print(Fore.BLUE + "Classification Result        :" + Back.WHITE + Fore.RED + " Scanty")
else : #TB Negative
  print(Fore.BLUE + "Classification Result        :" + Back.WHITE + Fore.RED + " Negative")