import numpy as np 
from sklearn import mixture
import os
import multiprocessing as mp
import time
import dispy
import sys
import socket
import logging
import matplotlib.pyplot as plt

main_dir = "/Users/Ahmed/Downloads/QSIURP2016/"
dataset_dir = main_dir + 'files/DataSets/'
files_dir = main_dir + 'files/'
results_dir = main_dir + 'results/'
def test():
	# out = open("GMM_2.gmm", 'wb')
	# # # pickle.dump(gmm, out)
	# out.close()
	# dispy_send_file("GMM_2.gmm")
	return 1
def fit((data, mixtureSize, filesInDataSetDir)):
	from sklearn import mixture
	# sys.stdout.write(data+'\n\n')
	gmm = mixture.GMM(n_components=mixtureSize,covariance_type='diag',min_covar=1.0,verbose=0)
	if filesInDataSetDir:
		d = '/Users/Ahmed/Downloads/QSIURP2016/files/DataSets/'
		fn = lambda x : x[1:]
	else:
		d = '/Users/Ahmed/Downloads/QSIURP2016/files/'
		fn = lambda x : x
	if type(data) == str:
		filename = data
		# return d+filename
		with open(d + filename,'r') as f:
			lines = f.readlines()
		# f.close()
		# while not f.isclosed:
		# 	f.close()
		# return f.closed
		lines = filter(lambda x : x.find("NaN") == -1 and x.find('nan') == -1, lines)
		# print filename
		lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
		data = map(fn, lines)
	# return data[0]
	# return map(len,data)
	gmm.fit(data)

	# out = open("/Users/Ahmed/Downloads/QSIURP2016/GMM_%d.gmm" % (mixtureSize), 'wb')
	# pickle.dump(gmm, out)
	# out.close()
	# dispy_send_file("/Users/Ahmed/Downloads/QSIURP2016/GMM_%d.gmm" % (mixtureSize))
	# return 1
	# gmm = mixture.GMM(n_components=2)
	return gmm

def score((gmm, data)):
	from sklearn import mixture
	return gmm.score(data)

def testFile((folder,f,GMMs,n)):
	import multiprocessing as mp
	import numpy as np 
	# return folder+f
	pool = mp.pool.ThreadPool()
	with open(folder+f, 'r') as f_feat:
		lines = f_feat.readlines()
	lines = filter(lambda x : x.find("NaN") == -1 and x.find('nan') == -1, lines)
	lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
	# f_feat.close()
	# print lines[0]
	# print folder,f,GMMs,n
	# return f_feat.closed
	# return GMMs[0].score(lines)
	# return map(len,lines)
	results = pool.map(score, map(lambda x : (x,lines),GMMs))
	# results = map(lambda x : x.score(lines), GMMs)
	# return len(results)
	pred = []
	for j in range(len(results[0])):
		res = map(lambda x : x[j], results)
		pred.append(np.argmax(res))
	counts = map(lambda x : pred.count(x), range(n))
	file_pred = np.argmax(counts)
	return file_pred

def testFolder((folder, GMMs, i, n, ext)):
	folder = main_dir + 'files/' + folder
	print folder
	correct_count = 0
	total = 0
	confMat = []
	for k in range(n):
		confMat.append([0]*n)

	files = os.listdir(folder)
	files = filter(lambda x : ext == x[-(len(ext)):], files)
	# if diff:
	# 	files = filter(lambda x : ".Dfeat" == x[-6:], files)	
	# else:
	# 	files = filter(lambda x : ".feat" in x, files)
	
	# print files
	jobs = map(lambda x1 : testFileCluster.submit(x1), map(lambda x : (folder, x, GMMs, n), files))
	
	res = map(lambda j : j(), jobs)
	# print res
	# print files
	for file_pred in res:
		confMat[i][file_pred] += 1.0
		if (file_pred == i):
			correct_count += 1.0
		total += 1
	return (confMat, correct_count, total)

# Submits training sets to the fitCluster to generate GMM's
def learn(mixtureSize, trainFiles, filesInDataSetDir=True):
	fitClusters = map(lambda x : dispy.JobCluster(fit,[x]), nodeList)
	# (mixtureSize, trainFiles) = t
	train_set = []
	jobs = []		
	for i in range(len(trainFiles)):
		print trainFiles[i]
		# gmm = mixture.GMM(n_components=mixtureSize[i],covariance_type='diag',min_covar=1.0,verbose=0)
		j = fitClusters[i % len(fitClusters)].submit((trainFiles[i], mixtureSize[i], filesInDataSetDir))
		# if (i % 2 == 0):
		# 	j = fitCluster1.submit((trainFiles[i], mixtureSize[i]))
		# else:
		# 	j = fitCluster2.submit((trainFiles[i], mixtureSize[i]))
		jobs.append(j)
	# print jobs
	GMMs = map(lambda job : job(), jobs)
	# print GMMs
	for f in fitClusters:
		f.close()
	return GMMs

# Given a set of folders in an order coresponding to the the order of GMMs (*) given classifies each 
# file in the folder and updates a confusion matrix. After classification is completed
# the confusion matrix is written to a file, named:
# 	
# 					prefix_[size(GMM[0]),...,size(GMM[n])]_[Class[0],...,Class[k]].txt
# 
# (*) 	For Example: testPerFile should be called with ([folder_A, Folder_B],[A,B],[GMM_A,GMM_B],_) 
# 		where folder_A is the folder that contains files of class A and GMM_A is the GMM trained on A.
def testPerFile(testFolders, GMMs, prefix, ext, f_res=None):
	classes = map(lambda x : (x[0].split('_'))[0], testFolders)
	# print GMMs
	if f_res == None:
		f_res = open("results/%s_%sG_%s.txt" % (prefix, str(map(lambda x : x.get_params()['n_components'],GMMs)), str(classes)), 'w')
	f_res.write(str(testFolders) + "\n")

	correct_count = 0
	total = 0
	confMat = []
	for i in range(len(testFolders)):
		confMat.append([0]*len(testFolders))

	jobs = []

	for i in range(len(testFolders)):
		# print testFolders[i]
		res = map(testFolder, map(lambda x : (x,GMMs,i,len(testFolders),ext), testFolders[i]))
		# print res
		confMats = map(lambda x : np.matrix(x[0]),res)
		correct_counts = map(lambda x : x[1],res)
		totals = map(lambda x : x[2],res)

		confMat += reduce(lambda x,y: x+y, confMats)
		correct_count += sum(correct_counts)
		total += sum(totals)

	f_res.write(str(correct_count / total) + "\n")
	f_res.write(str(classes) + "\n")
	for i in range(len(confMat)):
		l = confMat[i]
		f_res.write(str(l) + "\n")
	print confMat
	print (correct_count / total)
	if prefix == "crossVal":
		return (correct_count / total)
	f_res.close()



# node1 is the local machine
node1 = dispy.NodeAllocate("0.0.0.0",port=9998)
# node 2 is a remote machine
node2 = dispy.NodeAllocate('10.33.49.230', port=9993)
node3 = dispy.NodeAllocate('10.33.49.17', port=9991)
node4 = dispy.NodeAllocate('10.33.49.32', port=9992)
nodeList = [node1,node2,node3,node4]
# testCluster = dispy.JobCluster(test, [node1,node2])

# fitCluster runs the computation for learning the GMMs.
# Takes training sets as inputs and returns the corresponding GMM's

# testFileCluster classifies files
testFileCluster = None

def learnAndTest(x, trainSet, testFiles, crossVal=False, f_res=None):
	global testFileCluster
	print "size:",x
	# fitCluster1 = dispy.JobCluster(fit,[node1])
	# fitCluster2 = dispy.JobCluster(fit,[node1])

	# pool = mp.Pool()
	start_time = time.time()
	# args = [([2**x,2**x],[dataset_dir+"LectureHall.train", dataset_dir+"Bathroom.train"])]
	GMMs = learn([2**x]*len(trainSet),trainSet)
	# print GMMs
	print x, ("--- %s seconds ---" % (time.time() - start_time))
	# fitCluster1.close()
	# fitCluster2.close()

	# testFileCluster classifies files
	testFileCluster = dispy.JobCluster(testFile,[node1], depends=[score])
	start_time = time.time()
	acc = 0

	diff = len(filter(lambda x : "DIFF" in x, trainSet)) != 0 or len(filter(lambda x : ".Dtrain" in x, trainSet)) != 0
	norm = len(filter(lambda x : "NORM" in x, trainSet)) != 0 or len(filter(lambda x : ".Ntrain" in x, trainSet)) != 0
	C_0 = len(filter(lambda x : "C0" in x, trainSet)) != 0 or len(filter(lambda x : ".Ctrain" in x, trainSet)) != 0
	if diff:
		ext = ".Dfeat"
	if norm:
		ext = 'Nfeat'
	if C_0:
		ext = 'Cfeat'
	else:
		ext = '.feat'
	# print diff
	if crossVal:
		acc = testPerFile(testFiles, GMMs, "crossVal", ext, f_res)
	elif diff:
		testPerFile(testFiles, GMMs, "perFile_noOverlap_20CCs_diff+-", ".Dfeat", f_res)
	elif norm:
		testPerFile(testFiles, GMMs, "perFile_noOverlap_20CCs_NORM", ".Nfeat", f_res)
	elif C_0:
		testPerFile(testFiles, GMMs, "perFile_noOverlap_20CCs_C0", ".Cfeat", f_res)
	else:
		testPerFile(testFiles, GMMs,"perFile_noOverlap_20CCs", '.feat', f_res)
	print x, ("--- %s seconds ---" % (time.time() - start_time))
	testFileCluster.close()
	print acc
	return acc

def doCrossValidation(x, diff=False, norm=False):
	files = os.listdir(dataset_dir)
	if diff:
		dic = {'BH' : filter(lambda x : (x.split('-')[0] == 'BH_CV') and (x.split('.')[1] == 'Dtrain'), files),
		 'SH' : filter(lambda x : (x.split('-')[0] == 'SH_CV') and (x.split('.')[1] == 'Dtrain'), files),
		 'BR' : filter(lambda x : (x.split('-')[0] == 'BR_CV') and (x.split('.')[1] == 'Dtrain'), files),
		 'P' : filter(lambda x : (x.split('-')[0] == 'P_CV') and (x.split('.')[1] == 'Dtrain'), files),
		 'O' : filter(lambda x : (x.split('-')[0] == 'O_CV') and (x.split('.')[1] == 'Dtrain'), files)}
		f_res = open("results/%s_%sG_%s.txt" % ('crossVal_DIFF', [2**x]*4, dic.keys()), 'a')
	if norm:
		dic = {'BH' : filter(lambda x : (x.split('-')[0] == 'BH_CV') and (x.split('.')[1] == 'Ntrain'), files),
		 'SH' : filter(lambda x : (x.split('-')[0] == 'SH_CV') and (x.split('.')[1] == 'Ntrain'), files),
		 'BR' : filter(lambda x : (x.split('-')[0] == 'BR_CV') and (x.split('.')[1] == 'Ntrain'), files),
		 'P' : filter(lambda x : (x.split('-')[0] == 'P_CV') and (x.split('.')[1] == 'Ntrain'), files),
		 'O' : filter(lambda x : (x.split('-')[0] == 'O_CV') and (x.split('.')[1] == 'Ntrain'), files)}
		f_res = open("results/%s_%sG_%s.txt" % ('crossVal_NORM', [2**x]*4, dic.keys()), 'a')
	else:
		dic = {'BH' : filter(lambda x : (x.split('-')[0] == 'BH_CV') and (x.split('.')[1] == 'train'), files),
		 'SH' : filter(lambda x : (x.split('-')[0] == 'SH_CV') and (x.split('.')[1] == 'train'), files),
		 'BR' : filter(lambda x : (x.split('-')[0] == 'BR_CV') and (x.split('.')[1] == 'train'), files),
		 'P' : filter(lambda x : (x.split('-')[0] == 'P_CV') and (x.split('.')[1] == 'train'), files),
		 'O' : filter(lambda x : (x.split('-')[0] == 'O_CV') and (x.split('.')[1] == 'train'), files)}
		# print dic
		f_res = open("results/%s_%sG_%s.txt" % ('crossVal', [2**x]*4, dic.keys()), 'a')

	acc = []
	for i in range(1,len(dic['BR'])):
		br = dic['BR'][i]
		for j in range(len(dic['SH'])):
			bh = dic['SH'][j] 
			for k in range(len(dic['O'])):
				o = dic['O'][k]
				for l in range(len(dic['P'])):
					p = dic['P'][l]
					# print [br,lh,o,p]
					# print map(lambda x : [(x.split('-')[1]).split('.')[0]], [br,lh,o,p])
					print "i=%s j=%s k=%s l=%s" % (i,j,k,l)
					# print [br,lh,o,p], map(lambda x : [(x.split('-')[1]).split('.')[0]+'/'], [br,lh,o,p])
					acc.append(learnAndTest(x,[br,bh,o,p],map(lambda x : [(x.split('-')[1]).split('.')[0]+'/'], [br,bh,o,p]), crossVal=True, f_res=f_res))
		break
	f_res.write('\n\navg accuracy: %d' % (sum(acc)/len(acc)))
	f_res.write('\n\nmax accuracy: %d' % (max(acc)))
	f_res.close()

def checkCrossVal(filename):
	f = open(results_dir + filename)
	lines = f.readlines()
	f.close()
	accs = map(float,filter(lambda x : '[' not in x, lines))
	print "maximum accuracy: %f" % (max(accs))
	print "minimum accuracy: %f" % (min(accs))
	print "average accuracy: %f" % (float(sum(accs))/len(accs))
	print "variance: %f" % np.var(accs)

	bins = [0]*20
	for acc in accs:
		bins[int(acc * 100) / 5] += 1
	print bins
	plt.bar(range(0,100,5),bins, 5)
	plt.ylabel("Number of Files")
	plt.xlabel("Accuracy (%)")
	plt.title(filename)
	plt.savefig(results_dir + filename[:-3] + '.png',format='png')


def makeSuperVectorDataSet(dic,mixtureSize):
	class_idx = 0
	gmm_dic = dic.copy()
	for c in gmm_dic:
		gmm_dic[c] = map(lambda x : x + '.Dfeat', gmm_dic[c])
	gmm_dic = map(lambda c : learn([mixtureSize]*len(gmm_dic[c]), gmm_dic[c]), gmm_dic)
		# break
	print gmm_dic
if __name__ == '__main__':
	# checkCrossVal("crossVal_NORM_[64, 64, 64, 64]G_['P', 'SH', 'BH', 'O', 'BR'].txt")
	# print doCrossValidation(6, norm=True)
	# print learn([4,4],["LectureHall.train","Bathroom.train"])	
	# print fit(("LectureHall.train",2))
	# pool = mp.pool.ThreadPool()
	# map(learnAndTest,[0,1])
	# makeSuperVectorDataSet(dic,1)
	# learnAndTest(0,["BR_CV-BR_FC1_H.train","BH_CV-BH_2152_H.train","O_CV-O_1008_H.train","P_CV-P_2012_H.train"],[["BR_FC1_H/"],["BH_2152_H/"], ["O_1008_H/"], ["P_2012_H/"]])		
	for i in range(8,9,2):
		learnAndTest(i, ["BR.train","SH.train","O.train","P.train"], [["BR_LB2_H/"],["SH_2147_H/"], ["O_1018_H/"], ["P_CS_H/"]])
# gmm = mixture.GMM(n_components=2)
# # print gmm.__getstate__()
# j = fitCluster.submit(([[1,2],[3,4]], 2))
# # j = testCluster.submit()
# gmm = j()
# print gmm
# j = scoreCluster.submit((gmm,[[4,5],[3,2]]))
# print j()