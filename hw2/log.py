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
learning_rate = 0.0001

tmp = np.isnan(test_data)
test_data[tmp] = 0
test_matrix = np.delete(test_data,0,0)

#feature
train_matrix = np.delete(train_data,0,0)

#get classes
y = which_class.T
w = np.zeros((1,106))
goodness_of_function = 0.0
bias = 0.0
l = [0,1,3,4,5]
train_matrix[:,l] = (train_matrix[:,l] - np.mean(train_matrix,axis = 0)[l])/np.std(train_matrix,axis = 0)[l]
goodness_of_function = -(np.dot(y,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias)))+ np.dot((-1)*y+1,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias))))

for iteration in range(500):
	wdotxall = sigmoid_function( np.dot(train_matrix, w.T) + bias)
	diff_array = -np.dot((y - wdotxall.T),train_matrix)/w.size
	bias_diff = - np.sum(y - wdotxall.T)/w.size
	#print (np.sum(y - wdotxall.T))
	print (iteration)

	bias = bias - learning_rate*bias_diff
	w = w - learning_rate*diff_array
	#print (bias)
	expect_ans = sigmoid_function(np.dot(train_matrix, w.T).T+bias)
	for x in range(w.size):
		if expect_ans[0][x] >= 0.5:
			expect_ans[0][x] = 1
		else:
			expect_ans[0][x] = 0
	accuracy = 0
	for x in range(w.size):
		if expect_ans[0][x] == y[x]:
			accuracy = accuracy + 1
	print (accuracy/w.size)

	goodness_of_function = -(np.dot(y,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias)))+ np.dot((-1)*y+1,np.log(sigmoid_function(np.dot(train_matrix,w.T) + bias))))
	
l = [0,1,3,4,5]

test_matrix[:,l] = (test_matrix[:,l] - np.mean(test_matrix,axis = 0)[l])/np.std(test_matrix,axis = 0)[l]

expnum = np.dot(test_matrix,w.T)+bias
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