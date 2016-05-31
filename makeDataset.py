import os
from random import shuffle
import math

d = 'files/'
files = os.listdir('files/')

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
				lines_fbe = f_fbe.readlines()
				
				lines_fbe = filter(lambda x : "NaN" not in x, lines_fbe)
				lines_fbe = map(lambda x : map(lambda x2 : str(math.log(float(x2) + 1e-6)), x.split(',')),lines_fbe)
				lines_fbe = map(lambda x1 : reduce(lambda x,y : x + "," + y, x1), lines_fbe)
				# shuffle(lines_fbe)
				# shuffle(lines_mfcc)
				f_mfcc.close()
				f_fbe.close()

				for i in range(len(lines_fbe)):
					f_feat.write(lines_mfcc[i][:-1] + ',' +lines_fbe[i][:-1] + '\n')
					f_data.write(lines_mfcc[i][:-1] + ',' + lines_fbe[i][:-1] + '\n')
				f_feat.close()
			f_data.close()

def makeDataSet(dic):
	class_idx = 0
	for c in dic:
		f_class_test = open(d+"DataSets/"+c+".test",'w')
		f_class_train = open(d+"DataSets/"+c+".train",'w')

		trainSetSize = len(dic[c]) * 0.5

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
			
makeFeatFiles()
dic = {'BiglectureHall':['1202','2152'],
	 'SmallLectureHall':['1030','1190','2052'],
	 'LectureHall' : ['1030','1202','2052','1190','2152'],
	 'Bathroom' : ["Bathroom_lockers","Bathroom2_locker"]}

makeDataSet(dic)
