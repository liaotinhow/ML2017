import numpy as np
from numpy import loadtxt
import os
import sys
f = open('ans_one.txt','w')
A = loadtxt(sys.argv[1], dtype = "int64", comments ="#", delimiter=",")
B = loadtxt(sys.argv[2], dtype = "int64", comments ="#", delimiter=",")
c = np.dot(A,B)
c = np.reshape(c,-1)
c = sorted(c)


for x in c:
	f.write(str(x))
	f.write("\n")
