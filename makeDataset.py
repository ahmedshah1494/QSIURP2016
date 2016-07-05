import os
from random import shuffle
import math
import numpy as np
from parallelGMM import learn

d = 'files/'
files = os.listdir('files/')
files = filter(lambda x : len(x.split('_')) < 3 or x.split('_')[2] != 'R', files)

def csv2svm(class_idx, line):
	line = line.split(',')
	line = map(lambda x : str(x+1) + ':' + line[x], range(len(line)))
	line = str(class_idx) + ' ' + reduce(lambda x,y: x + ' ' + y, line)
	return line

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
def makeQuantFeatFiles(filterBy=None, diff=False, normalize=False, C_0=False):
	if normalize:
		quant_ext = ".Nquant"
	elif diff:
		quant_ext = ".Dquant"
	elif C_0:
		quant_ext = ".Cquant"
	else:
		quant_ext = ".quant"
	files = os.listdir('files/')
	if filterBy != None:
		files = filter(filterBy,files)
	for f in files:
		if os.path.isdir(d+f):
			os.system('cat files/%s/*%s > files/%s' % (f,quant_ext,f+quant_ext))

			dataFiles = os.listdir(d+f)
			dataFiles = filter(lambda x : quant_ext in x, dataFiles)

			f_data = open(d+f+quant_ext+'.svm', 'w')

			for dataFile in dataFiles:
				f_quant = open(d+f+'/'+dataFile, 'r')
				lines = f_quant.readlines()
				lines = map(lambda x : csv2svm(1,x), lines)
				for line in lines:
					f_data.write(line)

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

def makeCrossValidationDataSet(dic,diff=False,normalize=False,quantized=True):
	class_idx = 0
	if normalize:
		feat_ext = ".Nfeat"
		train_ext = ".Ntrain"
		if quantized:
			feat_ext = ".Nquant"
			train_ext = ".Nquant.svm.train"
	elif diff:
		feat_ext = ".Dfeat"
		train_ext = ".Dtrain"
		if quantized:
			feat_ext = ".Dquant"
			train_ext = ".Dquant.svm.train"
	else:
		feat_ext = ".feat"
		train_ext = ".train"
		if quantized:
			feat_ext = ".quant"
			train_ext = ".quant.svm.train"

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
					if quantized:
						f_class_train.write(line)
					else:
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


def makeCentroidDataSet(dic, normalize=False, diff=False, C_0=False):
	dataset_dir = 'files/DataSets/'
	if normalize:
		quant_ext = ".Nquant"
	elif diff:
		quant_ext = ".Dquant"
	elif C_0:
		quant_ext = ".Cquant"
	else:
		quant_ext = ".quant"

	classes = dic.keys()
	pairs = []
	for i in range(len(classes)):
		for j in range(i+1, len(classes)):
			if i == j:
				continue
			pairs.append((classes[i], classes[j]))
	

	for p in pairs:
		print p
		f_train = open('%s%sv%s%s.svm.train' % (dataset_dir,p[0],p[1],quant_ext), 'w')
		f_test = open('%s%sv%s%s.svm.test' % (dataset_dir,p[0],p[1],quant_ext), 'w')
		for j in range(len(p)):
			c = p[j]
			# train_files = dic[c][: int(0.8 * len(dic[c]))]
			Ntrain = int(0.8 * len(dic[c]))
			for i in range(len(dic[c])):
				f_quant = open('files/%s%s' % (dic[c][i],quant_ext), 'r')
				lines = f_quant.readlines()
				lines = map(lambda x : csv2svm((-1)**j,x), lines)
				if i < Ntrain:
					print "training on ", dic[c][i]
					for line in lines:
						# f_train.write(str((-1) ** j)+','+line)
						f_train.write(line)
				else:
					print "testing on ", dic[c][i]
					for line in lines:
						# f_test.write(str((-1) ** j)+','+line)
						f_test.write(line)
				f_quant.close()
		f_test.close()
		f_train.close()

def makeMultiClassCentroidDataSet(dic, normalize=False, diff=False, C_0=False):
	dataset_dir = 'files/DataSets/'
	if normalize:
		quant_ext = ".Nquant"
	elif diff:
		quant_ext = ".Dquant"
	elif C_0:
		quant_ext = ".Cquant"
	else:
		quant_ext = ".quant"

	class_idx = 0
	f_train = open(dataset_dir+'multiclass.train','w')
	for c in dic:
		if c == 'SH':
			continue
		Ntrain = int(0.8 * len(dic[c]))
		for i in range(len(dic[c])):
			f_quant = open('files/%s%s' % (dic[c][i],quant_ext), 'r')
			lines = f_quant.readlines()
			lines = map(lambda x : csv2svm(class_idx,x), lines)
			if i < Ntrain:
				print "training on ", dic[c][i]
				for line in lines:
					# f_train.write(str((-1) ** j)+','+line)
					f_train.write(line)
			else:
				break
		class_idx += 1

def uniquify(l):
	new_l = []
	for e in l:
		if e not in new_l:
			new_l.append(e)
	return new_l

def makePhysicalFeatureDataSet(physicalDic, dic, normalize=False, diff=False, C_0=False):
	dataset_dir = 'files/DataSets/'
	if normalize:
		quant_ext = ".Nquant"
	elif diff:
		quant_ext = ".Dquant"
	elif C_0:
		quant_ext = ".Cquant"
	else:
		quant_ext = ".quant"

	classes = physicalDic.keys()

	folders = uniquify([item for sublist in physicalDic.values() for item in sublist])
	trainFolders = []
	testFolders = []
	for c in dic:
		# shuffle(dic[c])
		Ntrain = int(0.8 * len(dic[c]))
		trainFolders += dic[c][:Ntrain]
		testFolders += dic[c][Ntrain:]
	
	for c in physicalDic:
		p = [physicalDic[c],filter(lambda x : x not in physicalDic[c], folders)]
		print p
		f_train = open('%sphysical/%s%s.svm.train' % (dataset_dir,c,quant_ext), 'w')
		f_test = open('%sphysical/%s%s.svm.test' % (dataset_dir,c,quant_ext), 'w')
		print '+1 :',len(p[0])
		print '-1 :',len(p[1])
		for j in range(len(p)):
			# train_files = physicaDic[c][: int(0.8 * len(physicaDic[c]))]
			for i in range(len(p[j])):
				f_quant = open('files/%s%s' % (p[j][i],quant_ext), 'r')
				lines = f_quant.readlines()
				lines = map(lambda x : csv2svm((-1)**j,x), lines)
				if p[j][i] in trainFolders:
					print "training on ", p[j][i]
					for line in lines:
						# f_train.write(str((-1) ** j)+','+line)
						f_train.write(line)
				else:
					print "testing on ", p[j][i]
					for line in lines:
						# f_test.write(str((-1) ** j)+','+line)
						f_test.write(line)
				f_quant.close()
		f_test.close()
		f_train.close()
		f_dataDis = open('%sphysical/%s.txt'% (dataset_dir,'dataDistribution'),'w')
		f_dataDis.write('trained on: %s\n' % (str(uniquify(trainFolders))))
		f_dataDis.write('tested on: %s\n' % (str(uniquify(testFolders))))
		f_dataDis.close()

def makeBinaryDataSet(dic, normalize=False, diff=False, C_0=False, crossVal=False):
	dataset_dir = 'files/DataSets/'
	if normalize:
		quant_ext = ".Nquant"
	elif diff:
		quant_ext = ".Dquant"
	elif C_0:
		quant_ext = ".Cquant"
	else:
		quant_ext = ".quant"

	trainFolders = []
	testFolders = []
	for c in dic:
		Ntrain = int(0.8 * len(dic[c]))
		trainFolders += dic[c][:Ntrain]
		testFolders += dic[c][Ntrain:]

	if crossVal:
		includeFolders = []
		for c in dic:
			l = []
			for i in range(len(dic[c])):
				l.append(dic[c][:])
				del l[i][i]
			includeFolders.append(l)
		while(len(includeFolders) > 1):
			includeFoldersNew = []
			for i in range(0,len(includeFolders),2):
				a = includeFolders[i]
				if i + 1 >= len(includeFolders):
					c = a
				else:
					b = includeFolders[i+1]
					c = []
					for j in range(len(a)):
						for k in range(len(b)):
							c.append(a[j] + b[k])
				includeFoldersNew.append(c)
			includeFolders = includeFoldersNew
		includeFolders = includeFolders[0]
	else:
		includeFolders = [trainFolders]

	for trainFolders in includeFolders:
		for i in range(len(dic.keys())):
			c = dic.keys()[i]
			if crossVal:
				f_train = open('%s%s_CV-%s.svm.train'% (dataset_dir+'/SVMcrossVal/', c, str(filter(lambda x : x not in trainFolders, sum(dic.values(),[])))),'w')
			else:
				f_train = open('%s%s.svm.train'% (dataset_dir, c),'w')
			for folder in dic[c]:
				if folder not in trainFolders:
					continue
				f_quant = open('files/%s%s' % (folder,quant_ext),'r')
				lines = f_quant.readlines()
				f_quant.close()
				lines = map(lambda x : csv2svm((1),x), lines)
				print "training on ", folder
				for line in lines:
					# f_train.write(str((-1) ** j)+','+line)
					f_train.write(line)
			for j in range(len(dic.keys())):
				if j == i:
					continue
				c = dic.keys()[j]
				for folder in dic[c]:
					if folder not in trainFolders:
						continue
					f_quant = open('files/%s%s' % (folder,quant_ext),'r')
					lines = f_quant.readlines()
					f_quant.close()
					lines = map(lambda x : csv2svm((-1),x), lines)
					print "training on ", folder
					for line in lines:
						f_train.write(line)
			f_train.close()

# def makeSVMDataSet():
# 	files = os.listdir('/files/DataSets/')
# 	files = filter(lambda x : "_GMMSV" in x, files)

# 	trainFiles = filter(lambda x : x[-5:] == 'train', files)
# 	testFiles = filter(lambda x : x[-5:] == 'test', files)

# 	train_f = open('svm.train')
# 	for fname in trainFiles:
if __name__ == '__main__':
	# makeDiffFeatFiles(lambda x : "_R" not in x)			
	# makeFeatFiles(lambda x : "2152" in x or "2147" in x or "1185" in x)
	# makeDiffFeatFiles(lambda x : "2152" in x or "2147" in x or "1185" in x)
	# makeFeatFiles(lambda x : "_R" not in x, C_0=True)
	# makeQuantFeatFiles(lambda x : "_R" not in x,normalize=True)
	files = os.listdir(d)
	dic = {'SH' : filter(lambda x : (x.split('_')[0] == 'SH') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
		 'BH' : filter(lambda x : (x.split('_')[0] == 'BH') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
		 'BR' : filter(lambda x : (x.split('_')[0] == 'BR') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
		 'P' : filter(lambda x : (x.split('_')[0] == 'P') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files),
		 'O' : filter(lambda x : (x.split('_')[0] == 'O') and os.path.isdir(d+x) and (x.split('_')[2] == 'H'), files)}

	physicalFeatureDic = {'carpet': ['BH','SH','O'],
						  'windows': ['BH','SH','O','P'],
						  'wall>15ft': ['BH'],
						  'wallFurniture': ['P','O'],
						  'ceramics': ['P','BR'],
						  'chairs>3': ['BH','SH'],
						  'chairs[0-3]': ['P','O'],
						  'tables>2': ['SH'],
						  'tables[0-2]': ['O','P'],
						  'longDesk': ['BH']}

	for f in physicalFeatureDic:
		physicalFeatureDic[f] = sum(map(lambda y: dic[y], physicalFeatureDic[f]),[])

	# test_dic = {"test": ['test','test',"test"]}
	# dic = {"LectureHall": ['1030','1202','2052','1190','2152'],
	# 		'Bathroom': ['Bathroom_lockers','Bathroom2_locker']}
	# print dic
	makeCrossValidationDataSet(dic,normalize=True,quantized=True)
	# makeUniversalDataSet(dic, normalize=True)
	# makeSuperVectorFiles(dic,4)
	# makeSuperVectorDataSet(dic)
	# makeDifferenceDataSet(dic)
	# makeDataSet(dic, C_0=True)
	# makeCentroidDataSet(dic, normalize=True)
	# makeMultiClassCentroidDataSet(dic, normalize=True)
	# makePhysicalFeatureDataSet(physicalFeatureDic,dic, normalize=True)
	# makeBinaryDataSet(dic, normalize=True, crossVal=True)
