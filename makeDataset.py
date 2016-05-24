import os
from random import shuffle
d = 'files/'
files = os.listdir('files/')

def makeFeatFiles():
	for f in files:
		if os.path.isdir(d+f):
			# print d+f
			dataFiles = os.listdir(d+f)
			wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
			f_test = open(d+f+".test",'w')
			f_train = open(d+f+".train",'w')
			for wav in wavs:
				f_feat = open(d+f+'/'+wav+".feat",'w')
				f_mfcc = open(d+f+'/'+wav+".mfcc", 'r')
				f_fbe = open(d+f+'/'+wav+".fbe",'r')
				lines_mfcc = f_mfcc.readlines()
				lines_fbe = f_fbe.readlines()
				shuffle(lines_fbe)
				shuffle(lines_mfcc)
				f_mfcc.close()
				f_fbe.close()

				testSetSize = len(lines_fbe) * 0.8
				for i in range(len(lines_fbe)):
					f_feat.write(lines_mfcc[i][:-1] + lines_fbe[i][:-1] + '\n')
					if i < testSetSize:
						f_train.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
					else:
						f_test.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
				f_feat.close()
			f_test.close()
			f_train.close()

def makeDataSet(dic):
	class_idx = 0
	for c in dic:
		f_class_test = open(d+"DataSets/"+c+".test",'w')
		f_class_train = open(d+"DataSets/"+c+".train",'w')

		for fname in dic[c]:
			f_test = open(d+fname+".test",'r')
			f_train = open(d+fname+'.train','r')
			test_lines = f_test.readlines()
			train_lines = f_train.readlines()
			f_test.close()
			f_train.close()

			for line in test_lines:
				f_class_test.write(str(class_idx)+","+line)

			for line in train_lines:
				f_class_train.write(str(class_idx)+","+line)

		class_idx += 1
			
makeFeatFiles()
dic = {'BiglectureHall':['1202','2152'],
	 'SmallLectureHall':['1030','1190','2052'],
	 'LectureHall' : ['1030','1190','2052','1202','2152'],
	 'Bathroom' : ["Bathroom_lockers","Bathroom2_locker"]}

makeDataSet(dic)
