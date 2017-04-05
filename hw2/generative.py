import csv
import sys
import numpy as np
from io import StringIO
from numpy import genfromtxt
import random
import math


def sigmoid_function(z):
	return 1/(1+np.exp(-z))
	

np.set_printoptions(precision=8,suppress=True,linewidth=1000,threshold=np.nan)
train_data = genfromtxt(sys.argv[3], delimiter=',')
which_class = genfromtxt(sys.argv[4], delimiter=',')
test_data = genfromtxt(sys.argv[5], delimiter = ',')

tmp = np.isnan(train_data)
train_data[tmp] = 0
train_matrix = train_data[1,0:]

tmp = np.isnan(test_data)
test_data[tmp] = 0
test_matrix = np.delete(test_data,0,0)
learning_rate = 0.0000001

#feature
train_matrix = np.delete(train_data,0,0)

#get classes
y = which_class.T
w = np.zeros((1,106))
goodness_of_function = 0.0
bias = 0.0
l = [0,1,3,4,5]

train_matrix[:,l] = (train_matrix[:,l] - np.mean(train_matrix,axis = 0)[l])/np.std(train_matrix,axis = 0)[l]
#goodness_of_function = -np.sum(np.dot(y,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias)))+ np.dot((-1)*y+1,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias))))
mu_class1 = np.zeros((1,106))
mu_class2 = np.zeros((1,106))
class1_num = 0
class2_num = 0
for x in range(train_matrix.shape[0]):
	if y[x] == 1:
		class1_num = class1_num + 1
		mu_class1 = mu_class1 + train_matrix[x]
	else:
		class2_num = class2_num + 1
		mu_class2 = mu_class2 + train_matrix[x]
mu_class1 = mu_class1/class1_num
mu_class2 = mu_class2/class2_num

sigma_2 = np.zeros((106,106))
sigma_1 = np.zeros((106,106))
print (np.dot((train_matrix[x] - mu_class1),(train_matrix[x] - mu_class1).T).shape)
for x in range(train_matrix.shape[0]):
	if y[x] == 1:
		sigma_1 = sigma_1 + np.dot((train_matrix[x] - mu_class1).T,(train_matrix[x] - mu_class1))
	else:
		sigma_2 = sigma_2 + np.dot((train_matrix[x] - mu_class2).T,(train_matrix[x] - mu_class2))
sigma_1 = sigma_1/class1_num
sigma_2 = sigma_2/class2_num
sigma = class1_num/(class2_num+class1_num)*sigma_1 + class2_num/(class2_num+class1_num)*sigma_2

expnum = np.dot (np.dot(train_matrix,np.linalg.inv(sigma)),(mu_class1 - mu_class2).T) - 0.5*np.dot(mu_class1,np.dot(np.linalg.inv(sigma),mu_class1.T))+0.5*np.dot(mu_class2,np.dot(np.linalg.inv(sigma),mu_class2.T)) + math.log(class1_num/class2_num)

expnum = sigmoid_function(expnum)
for x in range(train_matrix.shape[0]):
		if expnum[x][0] >= 0.5:
			expnum[x][0] = 1
		else:
			expnum[x][0] = 0
accuracy = 0
for x in range(train_matrix.shape[0]):
	if expnum[x][0] == y[x]:
		accuracy = accuracy + 1
print (accuracy/train_matrix.shape[0])
l = [0,1,3,4,5]

test_matrix[:,l] = (test_matrix[:,l] - np.mean(test_matrix,axis = 0)[l])/np.std(test_matrix,axis = 0)[l]

expnum = np.dot (np.dot(test_matrix,np.linalg.inv(sigma)),(mu_class1 - mu_class2).T) - 0.5*np.dot(mu_class1,np.dot(np.linalg.inv(sigma),mu_class1.T))+0.5*np.dot(mu_class2,np.dot(np.linalg.inv(sigma),mu_class2.T)) + math.log(class1_num/class2_num)
print (expnum.shape)
expnum = sigmoid_function(expnum)
for x in range(test_matrix.shape[0]):
		if expnum[x][0] >= 0.5:
			expnum[x][0] = 1
		else:
			expnum[x][0] = 0
with open(sys.argv[6], 'w') as csvfile:
	fieldnames = ['id', 'label']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for x in range(test_matrix.shape[0]):
		id_x = str(x+1)
		writer.writerow({'id': id_x, 'label': int(round(expnum[x][0]))})