import os
from random import shuffle
import math
import numpy as np
from parallelGMM import learn

d = 'files/'
files = os.listdir('files/')
files = filter(lambda x : len(x.split('_')) < 3 or x.split('_')[2] != 'R', files)

def makeFeatFiles():
	for f in files:
		if os.path.isdir(d+f):
			# print d+f
			dataFiles = os.listdir(d+f)
			wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
			f_data = open(d+f+".feat",'w')

			for j in range(len(wavs)):
				wav = wavs[j]
				# print wav
				f_feat = open(d+f+'/'+wav+".feat",'w')
				f_mfcc = open(d+f+'/'+wav+".mfcc", 'r')
				f_fbe = open(d+f+'/'+wav+".fbe",'r')
				lines_mfcc = f_mfcc.readlines()
				lines_mfcc = filter(lambda x : 'NaN' not in x and 'nan' not in x, lines_mfcc)
				lines_fbe = f_fbe.readlines()
				
				# lines_fbe = filter(lambda x : "NaN" not in x, lines_fbe)
				# lines_fbe = map(lambda x : map(lambda x2 : str(math.log(float(x2) + 1e-6)), x.split(',')),lines_fbe)
				# lines_fbe = map(lambda x1 : reduce(lambda x,y : x + "," + y, x1), lines_fbe)
				# shuffle(lines_fbe)
				# shuffle(lines_mfcc)
				f_mfcc.close()
				# f_fbe.close()

				for i in range(len(lines_mfcc)):
					# f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
					# f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
					f_feat.write(lines_mfcc[i][:-1]+'\n')
					f_data.write(lines_mfcc[i][:-1]+'\n')
				f_feat.close()
			f_data.close()

def makeDiffFeatFiles():
	for f in files:
		if os.path.isdir(d+f):
			# print d+f
			dataFiles = os.listdir(d+f)
			wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
			f_data = open(d+f+".Dfeat",'w')

			for j in range(len(wavs)):
				wav = wavs[j]
				# print wav
				f_feat = open(d+f+'/'+wav+".Dfeat",'w')
				f_mfcc = open(d+f+'/'+wav+".mfcc", 'r')
				lines_mfcc = f_mfcc.readlines()
				lines_mfcc = filter(lambda x : 'NaN' not in x and 'nan' not in x, lines_mfcc)
				lines_mfcc_f = map(lambda x : np.matrix(map(lambda x2 : float(x2), x.split(','))), lines_mfcc)

				diff_lines = []
				for j in range(len(lines_mfcc)):
					if j == 0:
						diff_lines.append(lines_mfcc[j])
						continue
					line = lines_mfcc_f[j] - lines_mfcc_f[j - 1]
					# print train_lines[j]
					# print train_lines[j-1]
					# print line
					line = list(line.getA1())
					line = reduce(lambda x,y: str(x)+','+str(y), line)
					diff_lines.append(line + "\n")
					
					f_mfcc.close()
				# f_fbe.close()

				for i in range(len(lines_mfcc)):
					# f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
					# f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
					f_feat.write(lines_mfcc[i][:-1] + ',' +diff_lines[i][:-1] + '\n')
					f_data.write(lines_mfcc[i][:-1] + ',' +diff_lines[i][:-1] + '\n')
				f_feat.close()
			f_data.close()

def makeDataSet(dic):
	class_idx = 0
	for c in dic:
		f_class_test = open(d+"DataSets/"+c+".test",'w')
		f_class_train = open(d+"DataSets/"+c+".train",'w')

		trainSetSize = int(len(dic[c]) * 0.8)

		for i in range(len(dic[c])):
			fname = dic[c][i]
			if i < trainSetSize:
				print "training on ", c, fname
				f_train = open(d+fname+'.feat','r')			
				train_lines = f_train.readlines()			
				f_train.close()

				for line in train_lines:
					f_class_train.write(str(class_idx)+","+line)

			else:
				print "testing on ", c, fname
				f_test = open(d+fname+".feat",'r')
				test_lines = f_test.readlines()
				f_test.close()

				for line in test_lines:
					f_class_test.write(str(class_idx)+","+line)

		class_idx += 1

def makeCrossValidationDataSet(dic,diff=False):
	class_idx = 0
	for c in dic:
		for r in dic[c]:
			if diff:
				f_class_train = open(d+"DataSets/%s_CV-%s.Dtrain" % (c,r),'w')
			else:	
				f_class_train = open(d+"DataSets/%s_CV-%s.train" % (c,r),'w')

			for i in range(len(dic[c])):
				fname = dic[c][i]
				if r == fname:
					continue
				print "training on ", c, fname
				if diff:
					f_train = open(d+fname+'.Dfeat','r')				
				else:
					f_train = open(d+fname+'.feat','r')			
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

def makeSuperVectorDataSet(dic):
	class_idx = 0
	gmm_dic = dic.copy()
	for c in gmm_dic:

		class_idx += 1
			
# makeFeatFiles()
# makeDiffFeatFiles()
files = os.listdir(d)
dic = {'LH' : filter(lambda x : (x.split('_')[0] == 'LH') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'BR' : filter(lambda x : (x.split('_')[0] == 'BR') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'P' : filter(lambda x : (x.split('_')[0] == 'P') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
	 'O' : filter(lambda x : (x.split('_')[0] == 'O') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files)}

test_dic = {"test": ['test','test',"test"]}
# dic = {"LectureHall": ['1030','1202','2052','1190','2152'],
# 		'Bathroom': ['Bathroom_lockers','Bathroom2_locker']}
# print dic
makeCrossValidationDataSet(dic,diff=True)
# makeDifferenceDataSet(dic)
# makeDataSet(dic)
