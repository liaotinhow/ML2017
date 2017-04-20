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
w = np.array([ 0.04589089 , 0.02285008 ,-0.15873412 , 0.04499233 , 0.38596754 ,-0.53185508, -0.00604638 , 1.16473451])

with open(sys.argv[3], 'w') as csvfile:
	fieldnames = ['id', 'value']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for x in range(round(test_data.shape[0]/18)):
		val = np.dot(test_data[x*18+9,3:],w.T)
		print (val)
		id_x = 'id_' + str(x)
		writer.writerow({'id': id_x, 'value': int(round(val))})