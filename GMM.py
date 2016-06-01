import numpy as np
from sklearn import mixture
import os
import multiprocessing as mp
import time
import dispy
import sys

def makeGMM(size, ts, q, i):
	# print ts
	print "starting worker:",os.getpid()
	gmm = mixture.GMM(n_components=size,covariance_type='diag',min_covar=1.0,verbose=0)
	gmm.fit(ts)
	q.put((i,gmm))
	print "stopping worker:",os.getpid()

def score((gmm,testSet)):
	return gmm.score(testSet)

def printMat(m):
	for l1 in m:
		print l1
files_folder = 'files/'
dataset_folder = 'files/DataSets/'
def learn((mixtureSize, trainFiles)):
	print "starting worker:",os.getpid()
	train_set = []
	
	for filename in trainFiles:
			f = open(filename,'r')
			lines = f.readlines()
			lines = filter(lambda x : x.find("NaN") == -1, lines)
			# print filename
			lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
			lines = map(lambda x : x[1:], lines)
			# train_set.append(lines)

	# print "files Read"
	# print train_set
	# GMMs = range(len(train_set))
	# GMMs = map(lambda i : (mixture.GMM(n_components=mixtureSizes[i],covariance_type='diag',min_covar=1.0,verbose=1),train_set[i]),GMMs)
	# GMMs = pool.map(makeGMM,GMMs,1)
	GMMs = []
	# for i in range(len(train_set)):
	# 	p = mp.Process(target=makeGMM, args=(mixtureSizes[i],train_set[i],worker_q,i))
	# 	p.start()
	# while(not worker_q.empty()):
	# 	GMMs.append(worker_q.get())
	# GMMs = sorted(GMMs, key=lambda x: x[0])
	# GMMs = map(lambda x : x[1], GMMs)
	# print "GMMMSMMSMSDnlk",len(GMMs)
	ts = lines
	gmm = mixture.GMM(n_components=mixtureSize,covariance_type='diag',min_covar=1.0,verbose=0)
	gmm.fit(ts)
	# GMMs.append(gmm)
	print "stopping worker:",os.getpid()
	# return GMMs
	return gmm

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
	pool = mp.pool.ThreadPool(processes=8)
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
	pool = mp.pool.ThreadPool(processes=8)
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

def testPerFile(testFolders, Classes, GMMs, prefix):
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

def learnAndTest(x):
	print "size:",x
	pool = mp.Pool(processes=8)
	# pool = 0
	# worker_q = mp.Queue()
	# main_q = mp.Queue()
	# for i in range(0,11,2):
	# 	GMMs = learn([(2**i)]*2, [dataset_folder+"LectureHall.train", dataset_folder+"Bathroom.train"], pool)
	# 	test([dataset_folder+"LectureHall.test", dataset_folder+"Bathroom.test"], GMMs, "noOverlap")
	# 	testPerFile([[files_folder+"1190/",files_folder+"2152/"],[files_folder+"Bathroom2_locker/"]], ["LectureHall","Bathroom"], GMMs, "perFile_noOverlap_diagCov")
	start_time = time.time()
	args = [(2**x,[dataset_folder+"LectureHall.train"]),(2**x,[dataset_folder+"Bathroom.train"])]
	GMMs = pool.map(learn,args)
	# print GMMs
	# m = mp.Process(target=learn, args=([64,64], [dataset_folder+"LectureHall.train", dataset_folder+"Bathroom.train"], pool))
	# m.start()
	# while(m.is_alive()):
	# 	continue
	# GMMs = learn([4,4], [dataset_folder+"LectureHall.train", dataset_folder+"Bathroom.train"], pool)
	print x, ("--- %s seconds ---" % (time.time() - start_time))
	start_time = time.time()
	testPerFile([[files_folder+"1190/",files_folder+"2152/"],[files_folder+"Bathroom2_locker/"]], ["LectureHall","Bathroom"], GMMs, "perFile_noOverlap")
	print x, ("--- %s seconds ---" % (time.time() - start_time))

def sqr(n):
	time.sleep(1)
	return n**2
if __name__ == '__main__':
	print 'main:',os.getpid()
	pool = mp.pool.ThreadPool()
	pool.map(learnAndTest,range(0,8,2))
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


















