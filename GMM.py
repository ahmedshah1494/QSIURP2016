import numpy as np
from sklearn import mixture
import os
import multiprocessing as mp
import time
import dispy
import sys

main_dir = "/Users/Ahmed/Downloads/QSIURP2016/"
dataset_dir = main_dir + 'files/DataSets/'
files_dir = main_dir + 'files/'

def makeGMM((size, ts)):
	# print ts
	print "starting worker:",os.getpid()
	gmm = mixture.GMM(n_components=size,covariance_type='diag',min_covar=1.0,verbose=0)
	gmm.fit(ts)
	print "stopping worker:",os.getpid()
	return gmm

def score((gmm,testSet)):
	return gmm.score(testSet)

def printMat(m):
	for l1 in m:
		print l1
files_folder = 'files/'
dataset_folder = 'files/DataSets/'

def learn(mixtureSize, trainFiles):
	ppool = mp.Pool()
	# train_set = []
	
	# for filename in trainFiles:
	# 		f = open(filename,'r')
	# 		lines = f.readlines()
	# 		lines = filter(lambda x : x.find("NaN") == -1, lines)
	# 		# print filename
	# 		lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
	# 		lines = map(lambda x : x[1:], lines)
	# 		# train_set.append(lines)
	# GMMs = []
	# ts = lines
	# # print map(len,ts)
	# gmm = mixture.GMM(n_components=mixtureSize,covariance_type='diag',min_covar=1.0,verbose=0)
	# gmm.fit(ts)
	# # GMMs.append(gmm)
	# print "stopping worker:",os.getpid()
	# # return GMMs
	# return gmm

	args = []
	GMMs = []
	for i in range(len(trainFiles)):
		filename = trainFiles[i]
		f = open(dataset_dir + filename,'r')
		lines = f.readlines()
		f.close()
		lines = filter(lambda x : x.find("NaN") == -1, lines)
		# print filename
		lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
		lines = map(lambda x : x[1:], lines)
		print trainFiles[i]
		args.append((mixtureSize[i],lines))
		# GMMs.append(mixture.GMM(n_components=mixtureSize[i],covariance_type='diag',min_covar=1.0,verbose=0))

	GMMs = ppool.map(makeGMM,args)
		
	# print jobs
	
	# print GMMs
	return GMMs

def test(testFiles, GMMs, prefix):
	test_set = []
	test_labels = []
	f_res = open("results/%s_%sG_%s.txt" % (prefix, str(map(lambda x : x.get_params()['n_components'],GMMs)), filter(lambda x : x.isupper() or x =='_', reduce(lambda y,z : y + '_' + z, testFiles))), 'w')
	f_res.write("classes: \n")
	for i in range(len(testFiles)):
			filename = testFiles[i]
			f_res.write("	%d. %s\n" % (i,filename))
			f = open(filename,'r')
			lines = f.readlines()
			lines = filter(lambda x : x.find("NaN") == -1, lines)
			# print filename
			lines = map(lambda x : map(lambda x2 : eval(x2), x.split(',')), lines)
			lines = map(lambda x : x[1:], lines)
			test_set += lines
			test_labels += [i] * len(lines)
	
	correct_count = 0
	
	results = map(lambda x : x.score(test_set), GMMs)
	confMat = []

	for i in range(len(testFiles)):
		confMat.append([0]*len(testFiles))

	for i in range(len(results[0])):
		res = map(lambda x : x[i], results)
		pred = np.argmax(res)

		confMat[test_labels[i]][pred] += 1

		if pred == test_labels[i]:
			correct_count += 1.0


	f_res.write(str(correct_count / len(test_set)) + "\n")
	for l in confMat:
		f_res.write(str(l) + "\n")

def testFile((folder,f,GMMs,n)):
	pool = mp.pool.ThreadPool()
	f_feat = open(folder+f, 'r')
	lines = f_feat.readlines()
	lines = filter(lambda x : x.find("NaN") == -1, lines)
	lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
	f_feat.close()

	results = pool.map(score, map(lambda x : (x,lines),GMMs))
	pred = []
	for j in range(len(results[0])):
		res = map(lambda x : x[j], results)
		pred.append(np.argmax(res))

	counts = map(lambda x : pred.count(x), range(n))
	file_pred = np.argmax(counts)
	return file_pred

def testFolder((folder, GMMs, i, n)):
	# sys.stdout.write("%s %s %s\n" % (folder, i, n))
	folder = files_dir + folder
	pool = mp.pool.ThreadPool()
	correct_count = 0
	total = 0
	confMat = []
	for k in range(n):
		confMat.append([0]*n)

	files = os.listdir(folder)
	files = filter(lambda x : ".feat" in x, files)

	res = pool.map(testFile,map(lambda x : (folder, x, GMMs, n), files))
	# sys.stdout.write("%s %s %s %s\n" % (folder, i, n, res))
	# for f in files:
		# print folder+f
		# f_feat = open(folder+f, 'r')
		# lines = f_feat.readlines()
		# lines = filter(lambda x : x.find("NaN") == -1, lines)
		# lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
		# f_feat.close()

		# results = map(score, map(lambda x : (x,lines),GMMs))
		# pred = []
		# for j in range(len(results[0])):
		# 	res = map(lambda x : x[j], results)
		# 	pred.append(np.argmax(res))

		# counts = map(lambda x : pred.count(x), range(len(testFolders)))
		# file_pred = np.argmax(counts)
		# # print folder+f, counts
	for file_pred in res:
		confMat[i][file_pred] += 1.0
		if (file_pred == i):
			correct_count += 1.0
		total += 1
	return (confMat, correct_count, total)

def testPerFile(testFolders, Classes, GMMs, prefix, f_res):
	if f_res == None:
		f_res = open("results/%s_%sG_%s.txt" % (prefix, str(map(lambda x : x.get_params()['n_components'],GMMs)), filter(lambda x : x.isupper() or x =='_', reduce(lambda y,z : y + '_' + z, Classes))), 'w')
	f_res.write("classes: \n")
	ppool = mp.Pool()

	correct_count = 0
	total = 0
	confMat = []
	for i in range(len(testFolders)):
		confMat.append([0]*len(testFolders))

	for i in range(len(testFolders)):
		res = ppool.map(testFolder, map(lambda x : (x,GMMs,i,len(testFolders)), testFolders[i]))
		confMats = map(lambda x : np.matrix(x[0]),res)
		correct_counts = map(lambda x : x[1],res)
		totals = map(lambda x : x[2],res)

		confMat += reduce(lambda x,y: x+y, confMats)
		correct_count += sum(correct_counts)
		total += sum(totals)
		# for folder in testFolders[i]:
			# files = os.listdir(folder)
			# files = filter(lambda x : ".feat" in x, files)
			# for f in files:
			# 	# print folder+f
			# 	f_feat = open(folder+f, 'r')
			# 	lines = f_feat.readlines()
			# 	lines = filter(lambda x : x.find("NaN") == -1, lines)
			# 	lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
			# 	f_feat.close()

			# 	results = pool.map(score, map(lambda x : (x,lines),GMMs))
			# 	pred = []
			# 	for j in range(len(results[0])):
			# 		res = map(lambda x : x[j], results)
			# 		pred.append(np.argmax(res))

			# 	counts = map(lambda x : pred.count(x), range(len(testFolders)))
			# 	file_pred = np.argmax(counts)
			# 	# print folder+f, counts
			# 	confMat[i][file_pred] += 1.0
			# 	if (file_pred == i):
			# 		correct_count += 1.0
			# 	total += 1

	f_res.write(str(correct_count / total) + "\n")
	for l in confMat:
		f_res.write(str(l) + "\n")
	f_res.close()

def learnAndTest(x, trainSet, testFiles, crossVal=False, f_res=None):
	print "size:",x
	

	# pool = mp.Pool()
	start_time = time.time()
	# args = [([2**x,2**x],[dataset_dir+"LectureHall.train", dataset_dir+"Bathroom.train"])]
	GMMs = learn([2**x]*4,trainSet)
	# print GMMs
	print x, ("--- %s seconds ---" % (time.time() - start_time))
	

	
	start_time = time.time()
	acc = 0
	if crossVal:
		acc = testPerFile(testFiles, ["Bathroom","LectureHall","Office","Pantry"], GMMs, "crossVal", f_res)
	else:
		testPerFile(testFiles, ["Bathroom","LectureHall","Office","Pantry"], GMMs, "TEST", f_res)
	print x, ("--- %s seconds ---" % (time.time() - start_time))
	# testFileCluster.close()
	# print acc
	# return acc

def doCrossValidation(x):
	files = os.listdir(dataset_dir)
	dic = {'LH' : filter(lambda x : (x.split('-')[0] == 'LH_CV') and (x.split('.')[1] == 'train'), files),
	 'BR' : filter(lambda x : (x.split('-')[0] == 'BR_CV') and (x.split('.')[1] == 'train'), files),
	 'P' : filter(lambda x : (x.split('-')[0] == 'P_CV') and (x.split('.')[1] == 'train'), files),
	 'O' : filter(lambda x : (x.split('-')[0] == 'O_CV') and (x.split('.')[1] == 'train'), files)}
	# print dic
	f_res = open("results/%s_%sG_%s2.txt" % ('crossVal', [2**x]*4, dic.keys()), 'a')
	acc = []
	for i in range(1,len(dic['BR'])):
		br = dic['BR'][i]
		for j in range(3,len(dic['LH'])):
			lh = dic['LH'][j] 
			for k in range(len(dic['O'])):
				o = dic['O'][k]
				for l in range(len(dic['P'])):
					p = dic['P'][l]
					# print [br,lh,o,p]
					# print map(lambda x : [(x.split('-')[1]).split('.')[0]], [br,lh,o,p])
					print "i=%s j=%s k=%s l=%s" % (i,j,k,l)
					acc.append(learnAndTest(x,[br,lh,o,p],map(lambda x : [(x.split('-')[1]).split('.')[0]+'/'], [br,lh,o,p]), crossVal=True, f_res=f_res))
		break
	f_res.write('\n\navg accuracy: %d' % (sum(acc)/len(acc)))
	f_res.write('\n\nmax accuracy: %d' % (max(acc)))
	f_res.close()
if __name__ == '__main__':
	print 'main:',os.getpid()
	pool = mp.pool.ThreadPool()
	# pool.map(learnAndTest,range(0,9,2))
	# learnAndTest(0,["BR_CV-BR_LB2_H.train","LH_CV-LH_2052_H.train","O_CV-O_1018_H.train","P_CV-P_CS_H.train"],[["BR_LB2_H/"],["LH_2052_H/"], ["O_1018_H/"], ["P_CS_H/"]])
	# node1 = dispy.NodeAllocate("0.0.0.0",port=1494)
	# node2 = dispy.NodeAllocate("0.0.0.0",port=1495)
	# node3 = dispy.NodeAllocate("0.0.0.0",port=1496)
	# node4 = dispy.NodeAllocate("0.0.0.0",port=1497)
	# cluster = dispy.JobCluster(sqr,nodes=[node1,node2,node3,node4])
	# jobs = []
	# for i in range(10):
	# 	print i
	# 	jobs.append(cluster.submit(i))
	# for job in jobs:
	# 	# print job.id
	# 	n = job()
	# 	print n

	# test([dataset_folder+"LectureHall.test", dataset_folder+"Bathroom.test"], GMMs, 'noOverlap')


















