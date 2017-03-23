import csv
import sys
import numpy as np
from io import StringIO
from numpy import genfromtxt
import random
np.set_printoptions(precision=8,suppress=True,linewidth=1000,threshold=np.nan)
train_data = genfromtxt(sys.argv[1], delimiter=',')
tmp = np.isnan(train_data)
train_data[tmp] = 0
train_matrix = train_data[1:19, 3:]
learning_rate = 0.00000001

test_data = genfromtxt(sys.argv[2], delimiter=',')

tmp = np.isnan(test_data)
test_data[tmp] = 0

for x in range(round((train_data.shape[0]-19)/18)):
	matrix = train_data[x*18+19:x*18+37, 3:]
	train_matrix = np.hstack((train_matrix,matrix))
#print (train_matrix.shape)

w = np.zeros((1,8))
 

goodness_of_function = 0.0
feature = np.zeros((5751,8))
bias = 0.0
model_ans = []
y = []

for x in range(train_matrix.shape[1] - 9):
	y.append(train_matrix[9][x+8])
	feature[x] = (train_matrix[9,x:x+8])

model_ans = np.dot(w,feature.T)
for x in range(train_matrix.shape[1] - 10):
	goodness_of_function = goodness_of_function + (model_ans[0][x] - y[x])**2
goodness_of_function =  (goodness_of_function/5751)**0.5
print (goodness_of_function)


#while goodness_of_function > 5.8:
for iteration in range(10000):
	print (iteration)
	diff_array = np.zeros(8)
	bias_diff = 0.0
	for x in range(500):
		which = random.randint(0,5750)
		diff_array = diff_array + 2*(y[which] - (bias + np.dot(w,np.reshape(feature[which],(8,1)))))*(-1)*(feature[which])
		bias_diff = bias_diff + 2*(y[which] - (bias + np.dot(w,np.reshape(feature[which],(8,1)))))*(-1)
	for x in range(round(test_data.shape[0]/18)):
		diff_array = diff_array + 2*(test_data[x*18+9,10] -(bias + np.dot(w,test_data[x*18+9,2:10])))*(-1)*(test_data[x*18+9,2:10])
		bias_diff = bias_diff + 2*(test_data[x*18+9,10] -(bias + np.dot(w,test_data[x*18+9,2:10])))*(-1)
	bias = bias - learning_rate*bias_diff
	w = w - learning_rate*diff_array
	goodness_of_function = 0.0
	model_ans = np.dot(w,feature.T)

	for n in range(train_matrix.shape[1] - 9):
		goodness_of_function = goodness_of_function + ((model_ans[0][n]+bias) - y[n])**2
	goodness_of_function =  (goodness_of_function/5751)**0.5
	print (goodness_of_function)



with open(sys.argv[3], 'w') as csvfile:
	fieldnames = ['id', 'value']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for x in range(round(test_data.shape[0]/18)):
		val = np.dot(test_data[x*18+9,3:],w.T)
		id_x = 'id_' + str(x)
		writer.writerow({'id': id_x, 'value': int(round(val[0]))})
