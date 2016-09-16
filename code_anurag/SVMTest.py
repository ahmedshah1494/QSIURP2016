import cPickle as pickle
import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import pairwise_kernels
import os
import sys

def test(fileList, SVMfile, label, resultsFolder):
	svm = load(open(SVMfile, 'r'))

	fl = open(fileList, 'r')
	lines = fl.readlines()
	fl.close()

	alldata = np.array([])
	for fname in lines:
		data = np.loadtxt(fname.strip(),delimiter=',')
		alldata = np.concatenate((alldata, data), axis=0)

	results = svm.predict(alldata)

	if not os.path.exists(resultsFolder):
		os.makedirs(resultsFolder)
	resfl = open(resultsFolder+"results.txt",'w')
	for i in range(len(results)):
		resfl.write("%s %d" % (line[i], results[i]))

if __name__ == '__main__':
	if len(sys.argv) >= 5:
		test(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])