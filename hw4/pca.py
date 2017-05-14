import csv
import sys
import numpy as np
from io import StringIO
from numpy import genfromtxt
import random
import math
import os
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import math
dataset = np.zeros((1,4096))

for x in range(ord('A'), ord('J')+1):
	for y in range(10):
		filename = chr(x) + str(0) + str(y) + ('.bmp')
		image = Image.open(filename)
		image = np.reshape(np.asarray( image ),-1)
		if chr(x) in 'A' and y == 0:
			dataset = image
		else:
			dataset = np.vstack((dataset,image))

dataset = dataset - np.mean(dataset,axis = 0)
u,s,v = np.linalg.svd(dataset)
fig = plt.figure(figsize=(64,64)) # 大小可自行決定

for x in range(9):
	pic = np.reshape(v[x],(64,64))
	ax = fig.add_subplot(3,3,x+1) # 每16個小圖一行
	ax.imshow(pic,cmap='gray') # image為某個filter的output或最能activate某個filter的input image
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.xlabel('hehe') # 如果你想在子圖下加小標的話
	plt.tight_layout()
fig.savefig('123.png')

fig = plt.figure(figsize=(64,64))
for x in range(100):
	pic = np.reshape(dataset[x],(64,64))
	ax = fig.add_subplot(10,10,x+1)
	ax.imshow(pic,cmap='gray') # image為某個filter的output或最能activate某個filter的input image
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.xlabel('hehe') # 如果你想在子圖下加小標的話
	plt.tight_layout()
fig.savefig('origin_data.png')

recover = np.zeros((100,5))
recover = np.dot(dataset,v[:5,:].T)
recover = np.mean(dataset,axis = 0) + np.dot(recover,v[:5,:])


fig = plt.figure(figsize=(64,64))
for x in range(100):
	pic = np.reshape(recover[x],(64,64))
	ax = fig.add_subplot(10,10,x+1)
	ax.imshow(pic,cmap='gray') # image為某個filter的output或最能activate某個filter的input image
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.xlabel('hehe') # 如果你想在子圖下加小標的話
	plt.tight_layout()
fig.savefig('recover_data.png')
minloss = 10
which = 0
for x in range(100):
	recover = np.dot(dataset,v[:x,:].T)
	recover = np.mean(dataset,axis = 0) + np.dot(recover,v[:x,:])
	error = math.sqrt(np.sum(np.square((dataset-recover)))/409600)/256
	if minloss > error:
		minloss = error
		which = x
	if error < 0.01:
		break
print (which)