import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import pairwise_kernels
import os
import dispy
import logging
import sys
sys.path.append('libsvm-3.21/tools')
import grid

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
def learn(filename):
	from sklearn.metrics.pairwise import pairwise_kernels
	from sklearn.svm import SVC

	svm = SVC(kernel = intersectionKernel)
	# (best_rate,best_params) = grid.find_parameters('files/DataSets/'+filename, options='-log2c -1,2,1 -log2g 1,1,1 -t 0')
	if type(filename) == str:
		data = parseSVM('files/DataSets/'+filename)
		# data = np.loadtxt('files/DataSets/'+filename, delimiter=',')
		# svm = SVC(C=best_params['c'],kernel=(lambda x,y: pairwise_kernels(x,y,metric=fast_hik,n_jobs=-1)))
		# svm = SVC(kernel=chi2_kernel)
		labels = map(lambda x : x[0], data)
		data = map(lambda x : x[1:],data)
	else:
		(data, labels) = filename
	svm.fit(data, labels)
	return svm

def test(filename, svm):
	from sklearn.svm import SVC
	import numpy as np
	# from sklearn.svm import SVC
	# return filename
	data = np.loadtxt('files/'+filename, delimiter=',')
	data = data.reshape(1, -1)

	# results = map(lambda x : x.predict(data)[0], svms)

	# for i in range(len(results)):
	# 	predLabel = svmLabels[i][(lambda x : 0 if x == 1 else 1)(results[i])]
	# print data
	# data = map(lambda x : x[1:],data)
	return svm.predict(data)
	# return predLabel

def label((filename, svmLabels)):
	main_dir = "/Users/Ahmed/Downloads/QSIURP2016/"
	import numpy as np
	from sklearn.svm import SVC

	data = np.loadtxt(main_dir+'files/'+filename, delimiter=',')
	data = data.reshape(1, -1)
	# return main_dir+'files/'+filename
	# with open(main_dir+'files/'+filename, 'r') as f_feat:
	# 		lines = f_feat.readlines()
	
	# lines = filter(lambda x : x.find("NaN") == -1 and x.find('nan') == -1, lines)
	# data = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)

	results = []
	return svms[0]
	for i in range(len(svms)):
		res = svms[i].predict(data)
		res = svmLabels[i][(lambda x : 0 if x == 1 else 1)(res[0])]
		results.append(res)

	# predLabels.append(results[np.argmax(map(results.count, results))])
	pred = results[np.argmax(map(results.count, results))]

	# print data
	# data = map(lambda x : x[1:],data)
	# return svm.predict(data)
	return pred

def doTests(trainFiles, testFolders):
	labels = ['BR','BH','O','P']
	svmLabels = map(lambda x : (x.split('.')[0].split('v')[0], x.split('.')[0].split('v')[1]), trainFiles)
	# print svmLabels
	svms = map(learn, trainFiles)
	correct_count = 0
	total = 0
	confMat = []
	for k in range(len(labels)):
		confMat.append([0]*len(labels))

	# node1 = dispy.NodeAllocate("0.0.0.0",port=9998)
	# # node 2 is a remote machine
	# node2 = dispy.NodeAllocate('10.33.49.230', port=9993)
	# node3 = dispy.NodeAllocate('10.33.49.17', port=9991)
	# node4 = dispy.NodeAllocate('10.33.49.32', port=9992)
	# nodeList = [node1,node2,node3,node4]

	# testCluster = dispy.JobCluster(label,[node1], loglevel=logging.DEBUG,depends=[fast_hik, intersectionKernel,svms])

	for folder in testFolders:
		print folder
		folderLabel = folder.split('_')[0]
		files = os.listdir('files/'+folder)
		files = filter(lambda x : x.split('.')[-1] == 'Nquant', files)

		# predLabels = map(lambda x: testCluster.submit((folder + x,svmLabels)), files[:2])
		# predLabels = map(lambda j : j(), predLabels)
		predLabels = []
		for f in files:
			results = []
			for i in range(len(svms)):
				res = test(folder + f,svms[i])
				res = svmLabels[i][(lambda x : 0 if x == 1 else 1)(res[0])]
				results.append(res)

			# predLabels.append(results[np.argmax(map(results.count, results))])
			pred = results[np.argmax(map(results.count, results))]
		# for pred in predLabels:
			# print pred
			confMat[labels.index(folderLabel)][labels.index(pred)] += 1
			if pred == folderLabel:
				correct_count += 1.0
			total += 1.0
			# print f, confMat
		print confMat 

	f_res = open("results/%s_%dD_%s.txt" % ('KMSVM_NORM', 1024, str(labels)), 'w')
	f_res.write(str(testFolders) + "\n")

	f_res.write(str(correct_count / total) + "\n")
	f_res.write(str(labels) + "\n")
	for i in range(len(confMat)):
		l = confMat[i]
		f_res.write(str(l) + "\n")
	print confMat
	
def doMultiClassTest(trainfile, testFolders):
	labels = ['P','BH','O','BR']
	svm = learn(trainfile)
	correct_count = 0
	total = 0
	confMat = []
	for k in range(len(labels)):
		confMat.append([0]*len(labels))
	for folder in testFolders:
		print folder
		folderLabel = folder.split('_')[0]
		files = os.listdir('files/'+folder)
		files = filter(lambda x : x.split('.')[-1] == 'Nquant', files)

		# predLabels = map(lambda x: testCluster.submit((folder + x,svmLabels)), files[:2])
		# predLabels = map(lambda j : j(), predLabels)
		predLabels = []
		for f in files:
			pred = labels[int(test(folder + f,svm)[0])]
			confMat[labels.index(folderLabel)][labels.index(pred)] += 1
			if pred == folderLabel:
				correct_count += 1.0
			total += 1.0
			# print f, confMat
		print confMat 

	f_res = open("results/%s_%dD_%s.txt" % ('KMSVM_NORM_MULTI', 1024, str(labels)), 'w')
	f_res.write(str(testFolders) + "\n")

	f_res.write(str(correct_count / total) + "\n")
	f_res.write(str(labels) + "\n")
	for i in range(len(confMat)):
		l = confMat[i]
		f_res.write(str(l) + "\n")
	print confMat

def getFeatureDict(allFeatures, positiveFs):
	binFeatureList = [-1] * len(allFeatures)
	for pf in positiveFs:
		if pf not in allFeatures:
			raise ValueError
		i = allFeatures.index(pf)
		binFeatureList[i] = 1
	return binFeatureList
def score(pred, actual):
	pred = np.matrix(pred)
	actual = np.matrix(actual)
	diff = pred - actual
	diff = diff.getA()[0]
	diff = map(abs,diff)
	return sum(diff)

def doPhysicalCharacteristicTest(trainFiles, testFolders, normalize=True):
	if normalize:
		quant_ext = 'Nquant'
	else:
		quant_ext = 'quant'

	features = map(lambda x : x.split('.')[0], trainFiles)
	# features = ['carpet', 'ceramics', 'chairs>3', 'chairs[0-3]', 'longDesk', 'tables>2', 'tables[0-2]', 'wall>15ft', 'wallFurniture', 'windows']
	featureDict = {'BR': getFeatureDict(features, ['ceramics']),
				   'BH': getFeatureDict(features, ['carpet','wall>15ft','longDesk','chairs>3']),
				   'SH': getFeatureDict(features, ['carpet','chairs>3','tables>2']),
				   'P': getFeatureDict(features, ['ceramics','chairs[0-3]','tables[0-2]','wallFurniture']),
				   'O': getFeatureDict(features, ['carpet','tables[0-2]','chairs[0-3]','wallFurniture'])}
	labels = map(lambda x : x.split('_')[0], testFolders)
	print labels
	correct_count = 0
	correct_count_phy = [0]*len(features)
	total = 0
	confMat = []
	for k in range(len(labels)):
		confMat.append([0]*len(labels))
	
	svms = map(lambda x: learn('physical/'+x), trainFiles)

	for folder in testFolders:
		print folder
		if 'P' in folder:
			print featureDict['P']
		folderLabel = folder.split('_')[0]
		files = os.listdir('files/'+folder)
		files = filter(lambda x : x.split('.')[-1] == 'Nquant', files)

		# predLabels = map(lambda x: testCluster.submit((folder + x,svmLabels)), files[:2])
		# predLabels = map(lambda j : j(), predLabels)
		predLabels = []
		for f in files:
			results = map(lambda x : test(folder + f, x)[0], svms)
			# results = featureDict['SH']
			actuals = [featureDict[c] for c in labels]
			scores = map(lambda x: score(results,x), actuals)
			if 'P' in folder:
				print f, results
			pred = np.argmin(scores)

			for i in range(len(features)):
				if results[i] == featureDict[folderLabel][i]:
					correct_count_phy[i] += 1.0

			confMat[labels.index(folderLabel)][pred] += 1
			if pred == labels.index(folderLabel):
				correct_count += 1.0
			total += 1.0
			# print f, confMat
		print confMat 

	f_res = open("results/%s_%dD_%s.txt" % ('KMSVM_NORM_PHYSICAL', 1024, str(labels)), 'w')
	f_res.write(str(testFolders) + "\n")

	f_res.write(str(correct_count / total) + "\n")
	f_res.write(str(labels) + "\n")
	for i in range(len(confMat)):
		l = confMat[i]
		f_res.write(str(l) + "\n")

	correct_count_phy = map(lambda x : x / total, correct_count_phy)
	for i in range(len(features)):
		f_res.write("%s : %f\n" % (features[i], correct_count_phy[i]))
	print confMat

def doBinaryTest(trainFiles, testFolders, normalize=True, f_res=None, svmLabels=None):
	labels = map(lambda x : x.split('_')[0], testFolders)
	if svmLabels == None:
		svmLabels = map(lambda x : x.split('.')[0], trainFiles)
		print trainFiles
	svms = map(learn, trainFiles)
	correct_count = 0
	none_count = 0
	total = 0
	confMat = []
	for k in range(len(labels)):
		confMat.append([0]*len(labels))

	# node1 = dispy.NodeAllocate("0.0.0.0",port=9998)
	# # node 2 is a remote machine
	# node2 = dispy.NodeAllocate('10.33.49.230', port=9993)
	# node3 = dispy.NodeAllocate('10.33.49.17', port=9991)
	# node4 = dispy.NodeAllocate('10.33.49.32', port=9992)
	# nodeList = [node1,node2,node3,node4]

	# testCluster = dispy.JobCluster(label,[node1], loglevel=logging.DEBUG,depends=[fast_hik, intersectionKernel,svms])

	for folder in testFolders:
		print folder
		folderLabel = folder.split('_')[0]
		files = os.listdir('files/'+folder)
		files = filter(lambda x : x.split('.')[-1] == 'Nquant', files)

		# predLabels = map(lambda x: testCluster.submit((folder + x,svmLabels)), files[:2])
		# predLabels = map(lambda j : j(), predLabels)
		predLabels = []
		for f in files:
			total += 1.0
			results = []
			pred = None
			results = map(lambda x : test(folder + f, x)[0], svms)
			# print results
			if 1 in results:
				pred = svmLabels[results.index(1)]

			# predLabels.append(results[np.argmax(map(results.count, results))])
			# pred = results[np.argmax(map(results.count, results))]
		# for pred in predLabels:
			# print pred
			if pred == None:
				none_count += 1
				continue
			confMat[labels.index(folderLabel)][labels.index(pred)] += 1
			if pred == folderLabel:
				correct_count += 1.0
			# print f, confMat
		print confMat 
	if f_res == None:
		f_res = open("results/%s_%dD_%s.txt" % ('KMSVM_NORM_BINARY_NoMiniBatch', 1024, str(labels)), 'w')
	f_res.write(str(testFolders) + "\n")

	f_res.write(str(correct_count / total) + "\n")
	f_res.write(str(none_count) + "\n")
	f_res.write(str(labels) + "\n")
	for i in range(len(confMat)):
		l = confMat[i]
		f_res.write(str(l) + "\n")
	print none_count
	print confMat

def doBinaryCrossVal(files):
	BH = filter(lambda x: "BH" in x, files)
	# SH = filter(lambda x: "SH" in x, files)
	BR = filter(lambda x: "BR" in x, files)
	O = filter(lambda x: "O" in x, files)
	P = filter(lambda x: "P" in x, files)

	f_res = open("results/KMSVM_NORM_BINARY_NoMiniBatch_CV-SH.txt",'a')
	for i in range(1,len(BR)):
		br = BR[i]
		for j in range(3,len(BH)):
			bh = BH[j]
			for k in range(len(O)):
				o = O[k]
				for l in range(len(P)):
					p = P[l]
					print ('i=%d j=%d k=%d l=%d' % (i,j,k,l))
					trainFiles = [br,bh,o,p]
					testFolders = map(lambda x : (x.split('-')[1]).split('.')[0] + '/', trainFiles)
					print testFolders

					trainData = []
					for positive in ((trainFiles)):
						trainLabels = []
						with open("files/DataSets/"+positive, 'r') as f:
							lines = f.readlines()
						data = map(lambda x: map(float, x.split(',')),lines)
						trainLabels += [1]*len(lines)
						for negative in ((trainFiles)):
							if positive == negative:
								continue
							with open("files/DataSets/"+negative, 'r') as f:
								lines = f.readlines()
							data += map(lambda x: map(float, x.split(',')),lines)
							trainLabels += [-1]*len(lines)
						trainData.append((data,trainLabels))
					doBinaryTest(trainData, testFolders, f_res=f_res, svmLabels=['BR','BH','O','P'])
			break
# svm = learn('PvBH.quant.svm.train')
# print test('P_2012_H.quant',svm)
# print doTests(filter(lambda x : 'Nquant.svm.train' in x and "SH" not in x, os.listdir('files/DataSets')),['BR_LB2_H/','BH_2152_H/', 'O_1018_H/', 'P_CS_H/'])
# doMultiClassTest('multiclass.train',['BR_LB2_H/','BH_2152_H/', 'O_1018_H/', 'P_CS_H/'])
# doPhysicalCharacteristicTest(filter(lambda x : 'Nquant.svm.train' in x and 'windows' not in x, os.listdir('files/DataSets/physical/')),['BR_LB2_H/','BH_2152_H/','O_1018_H/', 'P_CS_H/'])
# print doBinaryTest(filter(lambda x : 'svm.train' in x and "SH" not in x and 'v' not in x.split('.')[0], os.listdir('files/DataSets')),['BR_LB2_H/','BH_2152_H/', 'O_1018_H/', 'P_CS_H/'])
doBinaryCrossVal(filter(lambda x : 'CV' in x and '.svm.train' in x, os.listdir('files/DataSets')))

