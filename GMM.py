import numpy as np
from sklearn import mixture
import os

def printMat(m):
	for l1 in m:
		print l1
files_folder = 'files/'
dataset_folder = 'files/DataSets/'
def learn(mixtureSizes, trainFiles):
	train_set = []
	GMMs = []
	for filename in trainFiles:
			f = open(filename,'r')
			lines = f.readlines()
			lines = filter(lambda x : x.find("NaN") == -1, lines)
			# print filename
			lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
			lines = map(lambda x : x[1:], lines)
			train_set.append(lines)

	# print "files Read"
	for i in range(len(train_set)):
		ts = train_set[i]
		gmm = mixture.GMM(n_components=mixtureSizes[i],covariance_type='full',min_covar=1.0,verbose=1)
		gmm.fit(ts)
		GMMs.append(gmm)

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

def testPerFile(testFolders, Classes, GMMs, prefix):
	f_res = open("results/%s_%sG_%s.txt" % (prefix, str(map(lambda x : x.get_params()['n_components'],GMMs)), filter(lambda x : x.isupper() or x =='_', reduce(lambda y,z : y + '_' + z, Classes))), 'w')
	f_res.write("classes: \n")

	correct_count = 0
	total = 0
	confMat = []
	for i in range(len(testFolders)):
		confMat.append([0]*len(testFolders))

	for i in range(len(testFolders)):
		for folder in testFolders[i]:			
			files = os.listdir(folder)
			files = filter(lambda x : ".feat" in x, files)
			for f in files:
				# print folder+f
				f_feat = open(folder+f, 'r')
				lines = f_feat.readlines()
				lines = filter(lambda x : x.find("NaN") == -1, lines)
				lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
				f_feat.close()

				results = map(lambda x : x.score(lines), GMMs)
				pred = []
				for j in range(len(results[0])):
					res = map(lambda x : x[j], results)
					pred.append(np.argmax(res))

				counts = map(lambda x : pred.count(x), range(len(testFolders)))
				file_pred = np.argmax(counts)
				# print folder+f, counts
				confMat[i][file_pred] += 1.0
				if (file_pred == i):
					correct_count += 1.0
				total += 1

	f_res.write(str(correct_count / total) + "\n")
	for l in confMat:
		f_res.write(str(l) + "\n")
	f_res.close()




# for i in range(0,11,2):
# 	GMMs = learn([(2**i)]*2, [dataset_folder+"LectureHall.train", dataset_folder+"Bathroom.train"])
# 	test([dataset_folder+"LectureHall.test", dataset_folder+"Bathroom.test"], GMMs, "noOverlap")

GMMs = learn([16,1], [dataset_folder+"LectureHall.train", dataset_folder+"Bathroom.train"])
# testPerFile([[files_folder+"1190/",files_folder+"2152/"],[files_folder+"Bathroom2_locker/"]], ["LectureHall","Bathroom"], GMMs, "perFile_noOverlap")
test([dataset_folder+"LectureHall.test", dataset_folder+"Bathroom.test"], GMMs, 'noOverlap')
