import os
from random import shuffle
import math
import numpy as np
from parallelGMM import learn

d = 'files/'
files = os.listdir('files/')
files = filter(lambda x : len(x.split('_')) < 3 or x.split('_')[2] != 'R', files)

def makeFeatFiles(filterBy=None, normalize=False, C_0=False):
	files = os.listdir('files/')
	if filterBy != None:
		files = filter(filterBy,files)
	for f in files:
		if os.path.isdir(d+f):
			# print d+f
			dataFiles = os.listdir(d+f)
			wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)

			if normalize:
				f_data = open(d+f+".Nfeat",'w')
			if C_0:
				f_data = open(d+f+".Cfeat",'w')
			else:
				f_data = open(d+f+".feat",'w')

			for j in range(len(wavs)):
				wav = wavs[j]
				# print wav
				
				f_mfcc = open(d+f+'/'+wav+".mfcc", 'r')
				# f_fbe = open(d+f+'/'+wav+".fbe",'r')
				lines_mfcc = f_mfcc.readlines()
				lines_mfcc = filter(lambda x : 'NaN' not in x and 'nan' not in x, lines_mfcc)
				# lines_fbe = f_fbe.readlines()
				
				# lines_fbe = filter(lambda x : "NaN" not in x, lines_fbe)
				# lines_fbe = map(lambda x : map(lambda x2 : str(math.log(float(x2) + 1e-6)), x.split(',')),lines_fbe)
				# lines_fbe = map(lambda x1 : reduce(lambda x,y : x + "," + y, x1), lines_fbe)
				# shuffle(lines_fbe)
				# shuffle(lines_mfcc)
				f_mfcc.close()
				# f_fbe.close()

				if normalize or C_0:					
					if normalize:
						lines_mfcc = map(lambda x : np.matrix(map(float,x.split(','))), lines_mfcc)
						means = map(lambda x1 : x1 / len(lines_mfcc),reduce(lambda x,y: x + y, lines_mfcc))[0]
						f_feat = open(d+f+'/'+wav+".Nfeat",'w')
					else:
						means = np.matrix([sum(map(lambda x : float(x.split(',')[0]), lines_mfcc)) / len(lines_mfcc)] * (lines_mfcc[0].count(',') + 1))
						lines_mfcc = map(lambda x : np.matrix(map(float,x.split(','))), lines_mfcc)
						f_feat = open(d+f+'/'+wav+".Cfeat",'w')

					lines_mfcc = map(lambda x : x - means, lines_mfcc)
					lines_mfcc = map(lambda x : map(str,x.getA1()),lines_mfcc)
					lines_mfcc = map(lambda x1 : reduce(lambda x,y : x + ',' + y, x1), lines_mfcc)

					for i in range(len(lines_mfcc)):
						# f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
						# f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
						f_feat.write(lines_mfcc[i][:-1]+'\n')
						f_data.write(lines_mfcc[i][:-1]+'\n')
				else:
					f_feat = open(d+f+'/'+wav+".feat",'w')
					for i in range(len(lines_mfcc)):
						# f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
						# f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
						f_feat.write(lines_mfcc[i][:-1]+'\n')
						f_data.write(lines_mfcc[i][:-1]+'\n')
				f_feat.close()
			f_data.close()

def makeDiffFeatFiles(filterBy=None):
	files = os.listdir('files/')
	if filterBy != None:
		files = filter(filterBy,files)
	for f in files:
		if os.path.isdir(d+f):
			# print d+f
			dataFiles = os.listdir(d+f)
			wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
			f_data = open(d+f+".Dfeat",'w')

			for j in range(len(wavs)):
				wav = wavs[j]
				print wav
				f_feat = open(d+f+'/'+wav+".Dfeat",'w')
				f_mfcc = open(d+f+'/'+wav+".mfcc", 'r')
				lines_mfcc = f_mfcc.readlines()
				lines_mfcc = filter(lambda x : 'NaN' not in x and 'nan' not in x, lines_mfcc)
				lines_mfcc_f = map(lambda x : np.matrix(map(lambda x2 : float(x2), x.split(','))), lines_mfcc)

				diff_lines = []
				for j in range(len(lines_mfcc)):
					lineP = lines_mfcc[j][:-1]
					lineN = lines_mfcc[j][:-1]

					if j > 0:
						lineP = lines_mfcc_f[j] - lines_mfcc_f[j - 1]
						lineP = list(lineP.getA1())
						lineP = reduce(lambda x,y: str(x)+','+str(y), lineP)

					if j < len(lines_mfcc) - 1:
						lineN = lines_mfcc_f[j] - lines_mfcc_f[j + 1]
						lineN = list(lineN.getA1())
						lineN = reduce(lambda x,y: str(x)+','+str(y), lineN)

					diff_lines.append(lineN + ',' + lineP + "\n")
					
					f_mfcc.close()
				# f_fbe.close()

				for i in range(len(lines_mfcc)):
					# f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
					# f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
					f_feat.write(lines_mfcc[i][:-1] + ',' +diff_lines[i][:-1] + '\n')
					f_data.write(lines_mfcc[i][:-1] + ',' +diff_lines[i][:-1] + '\n')
				f_feat.close()
			f_data.close()

def makeDataSet(dic, normalize=False, C_0=False):
	class_idx = 0
	if normalize:
		feat_ext = ".Nfeat"
		test_ext = "_NORM.test"
		train_ext = "_NORM.train"
	elif C_0:
		feat_ext = ".Cfeat"
		test_ext = "_C0.test"
		train_ext = "_C0.train"
	else:
		feat_ext = ".feat"
		test_ext = ".test"
		train_ext = ".train"
	for c in dic:
		f_class_test = open(d+"DataSets/"+c+test_ext,'w')
		f_class_train = open(d+"DataSets/"+c+train_ext,'w')

		trainSetSize = int(len(dic[c]) * 0.8)
		# dic[c] = np.random.shuffle(dic[c])
		for i in range(len(dic[c])):
			fname = dic[c][i]
			if i < trainSetSize:
				print "training on ", c, fname
				f_train = open(d+fname+feat_ext,'r')			
				train_lines = f_train.readlines()			
				f_train.close()

				for line in train_lines:
					f_class_train.write(str(class_idx)+","+line)

			else:
				print "testing on ", c, fname
				f_test = open(d+fname+feat_ext,'r')
				test_lines = f_test.readlines()
				f_test.close()

				for line in test_lines:
					f_class_test.write(str(class_idx)+","+line)

		class_idx += 1

def makeCrossValidationDataSet(dic,diff=False,normalize=False):
	class_idx = 0
	if normalize:
		feat_ext = ".Nfeat"
		train_ext = ".Ntrain"
	elif diff:
		feat_ext = ".Dfeat"
		train_ext = ".Dtrain"
	else:
		feat_ext = ".feat"
		train_ext = ".train"
	for c in dic:
		for r in dic[c]:
			
			f_class_train = open(d+"DataSets/%s_CV-%s%s" % (c,r,train_ext),'w')
			

			for i in range(len(dic[c])):
				fname = dic[c][i]
				if r == fname:
					continue
				print "training on ", c, fname
				
				f_train = open(d+fname+feat_ext,'r')				
				
				train_lines = f_train.readlines()			
				f_train.close()

				for line in train_lines:
					f_class_train.write(str(class_idx)+","+line)

		class_idx += 1

def makeDifferenceDataSet(dic):
	class_idx = 0
	for c in dic:
		f_class_train = open(d+"DataSets/%s_DIFF.train" % (c),'w')
		f_class_test = open(d+"DataSets/%s_DIFF.test" % (c),'w')

		trainSetSize = int(len(dic[c]) * 0.8)

		# dic[c] = np.random.shuffle(dic[c])
		for i in range(len(dic[c])):
			fname = dic[c][i]
			f_train = open(d+fname+'.Dfeat','r')			
			train_lines = f_train.readlines()
			f_train.close()

			if i < trainSetSize:
				print "training on ", c, fname
			else:
				print "testing on ", c, fname
			for line in train_lines:
				if i < trainSetSize:
					f_class_train.write(str(class_idx)+","+line)
				else:
					f_class_test.write(str(class_idx)+","+line)
			# else:
			# 	print "testing on ", c, fname
			# 	f_test = open(d+fname+".feat",'r')
			# 	test_lines = f_test.readlines()
			# 	test_lines = filter(lambda x : x.find("NaN") == -1, test_lines)
			# 	test_lines = map(lambda x : np.matrix(map(lambda x2 : float(x2), x.split(','))), test_lines)			
			# 	f_test.close()

			# 	diff_lines = []
			# 	for j in range(len(train_lines)):
			# 		if j == 0:
			# 			continue
			# 		line = test_lines[j] - test_lines[j - 1]
			# 		line = list(line.getA1())
			# 		line = reduce(lambda x,y: str(x)+','+str(y), line)
			# 		diff_lines.append(line)

			# 	for line in diff_lines:
			# 		f_class_test.write(str(class_idx)+","+line)

		class_idx += 1

def makeSuperVectorFiles(dic,mixtureSize):
	class_idx = 0
	for c in dic:
		for folder in dic[c]:
			files = os.listdir(d+folder)
			files = filter(lambda x : ".mfcc" == x[-5:], files)
			files = map(lambda x: folder+'/'+x, files)
			GMMs = learn([mixtureSize]*len(files), files, False)
			# print GMMs
			# print files
			fl_gmm = open(d+folder+'.gmm','w')
			for i in range(len(files)):
				f = files[i]
				print f
				f_gmm = open(d+f+'.gmm','w')
				gmm = GMMs[i]
				if gmm == None:
					print "=========================NONE============================"
					continue
				line = str(class_idx) + ' '

				means = gmm.means_
				weights = gmm.weights_

				weights_str = map(str,weights)
				weights_str = map(lambda x : str(x)+':'+ weights_str[x-1],range(1,len(weights_str)+1)) 
				weights_str = reduce(lambda x,y: str(x) + ' ' + str(y), weights_str)
				line += weights_str + " "
				means_str = map(lambda x : map(str, x), means)
				means_str = reduce(lambda x,y: y + x,means_str)
				# print means_str, len(means_str)
				means_str = map(lambda x : str(x)+':'+ means_str[x-1-mixtureSize],range(mixtureSize + 1,len(means_str) + mixtureSize + 1))
				# print means_str
				line += reduce(lambda x,y: str(x) + ' ' + str(y),means_str) + ' '
				f_gmm.write(line+'\n')
				fl_gmm.write(line+'\n')
				f_gmm.close()
			fl_gmm.close()
			# return

		class_idx += 1

def makeSuperVectorDataSet(dic):
	for c in dic:
		f_class_train = open(d+"DataSets/%s_GMMSV.train" % (c),'w')
		f_class_test = open(d+"DataSets/%s_GMMSV.test" % (c),'w')

		trainSetSize = int(len(dic[c]) * 0.8)

		for i in range(len(dic[c])):
			fname = dic[c][i]
			f_train = open(d+fname+'.gmm','r')			
			train_lines = f_train.readlines()
			f_train.close()

			if i < trainSetSize:
				print "training on ", c, fname
			else:
				print "testing on ", c, fname
			for line in train_lines:
				if i < trainSetSize:
					f_class_train.write(line)
				else:
					f_class_test.write(line)

def makeUniversalDataSet(dic, diff=False, normalize=False, C_0=False):
	if normalize:
		feat_ext = ".Nfeat"
		train_ext = ".Ntrain"
	elif diff:
		feat_ext = ".Dfeat"
		train_ext = ".Dtrain"
	else:
		feat_ext = ".feat"
		train_ext = ".train"

	folders = dic.values()
	folders = (reduce(lambda x,y : x + y, folders))

	folders = map(lambda x : d + x + feat_ext, folders)

	os.system('cat %s > %s' % (reduce(lambda x,y : x + ' ' + y, folders), d+'DataSets/universal'+train_ext))
	

# def makeSVMDataSet():
# 	files = os.listdir('/files/DataSets/')
# 	files = filter(lambda x : "_GMMSV" in x, files)

# 	trainFiles = filter(lambda x : x[-5:] == 'train', files)
# 	testFiles = filter(lambda x : x[-5:] == 'test', files)

# 	train_f = open('svm.train')
# 	for fname in trainFiles:

# makeDiffFeatFiles(lambda x : "_R" not in x)			
# makeFeatFiles(lambda x : "2152" in x or "2147" in x or "1185" in x)
# makeDiffFeatFiles(lambda x : "2152" in x or "2147" in x or "1185" in x)
# makeFeatFiles(lambda x : "_R" not in x, C_0=True)
files = os.listdir(d)
dic = {'SH' : filter(lambda x : (x.split('_')[0] == 'SH') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'BH' : filter(lambda x : (x.split('_')[0] == 'BH') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'BR' : filter(lambda x : (x.split('_')[0] == 'BR') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'P' : filter(lambda x : (x.split('_')[0] == 'P') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'O' : filter(lambda x : (x.split('_')[0] == 'O') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files)}

# test_dic = {"test": ['test','test',"test"]}
# dic = {"LectureHall": ['1030','1202','2052','1190','2152'],
# 		'Bathroom': ['Bathroom_lockers','Bathroom2_locker']}
# print dic
# makeCrossValidationDataSet(dic,normalize=True)
makeUniversalDataSet(dic, normalize=True)
# makeSuperVectorFiles(dic,4)
# makeSuperVectorDataSet(dic)
# makeDifferenceDataSet(dic)
# makeDataSet(dic, C_0=True)
