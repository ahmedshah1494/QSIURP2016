import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import pairwise_kernels
import os
import sys
import grid
import matplotlib.pyplot as plt
import cPickle as pickle


def parseSVM(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
	lines = map(lambda x: map(lambda y: y.split(':')[-1],x.split()), lines)
	lines = map(lambda x : map(float,x),lines)
	return lines

def fast_hik(x, y):
	return np.minimum(x, y).sum()

def intersectionKernel(x,y):
	return pairwise_kernels(x,y,metric=fast_hik,n_jobs=-1)
def learn(fileList, SVMdir, Kernel="linear"):
	fl = open(fileList, 'r')
	lines = fl.readlines()
	fl.close()

	allData = []
	labels = []
	for l in lines:
		[label, fname] = l.split()
		labels.append(int(label))
		fl = open(fname,'r')
		dataStr = fl.readlines()
		fl.close()
		data = map(float, dataStr)
		alldata.append(data)

	if Kernel="intersection":
		Kernel = intersectionKernel
	svm = SVC(kernel = Kernel)
	svm.fit(alldata, labels)
	pickle.dump(svm, open(SVMdir+"SVM.pkl", 'wb'))

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "-arg1 inFileList -arg2 SVMdir [-arg3 kernel]"
	if len(sys.argv) == 3:
		learn(sys.argv[1], sys.argv[2])
	if len(sys.argv) == 4:
		learn(sys.argv[1], sys.argv[2], kernel=sys.argv[3])