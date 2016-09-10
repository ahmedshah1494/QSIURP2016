import numpy as np
import os

def getAccuracyOverTime(roomType,nComps,nFolds,interval):
	# interval in seconds
	probsByTime = {}
	for f in range(nFolds):
		path_P="results/%s/fold_%d/%d/P.txt" % (roomType,f,nComps)
		if not os.path.exists("results/%s/analysis/fold_%d/%d/" % (roomType,f,nComps)):
			os.makedirs("results/%s/analysis/fold_%d/%d/"  % (roomType,f,nComps))
		resultPath = "results/%s/analysis/fold_%d/%d/result.txt" % (roomType,f,nComps)
		fl = open(path_P, 'r')
		resultFl = open(resultPath, 'w')
		lines = fl.readlines()
		lines = map(lambda x : (map(float,x.split())[1:]), lines)
		
		labels = map(np.argmax,lines)
		linesPerInterval = interval/3
		for i in range(linesPerInterval,len(labels),linesPerInterval):
			prob = (sum(labels[:i])/float(i));
			resultFl.write("%d %f\n" % (i*3, prob))
			if probsByTime.get(i*3) == None:
				probsByTime[i*3] = 0.0
			probsByTime[i*3] += prob
			# total = sum(map(np.array, lines[:i]))
			# if prevProb == 1:
			# 	prob = total[1]
			# else:
			# 	prob = (total[1]/prevProb)
			# prevProb = total[1]
			# print prob
			# # print total, sum(map(sum,lines[:i]))

		# path_N="results/%s/fold_%d/%d/N.txt" % (roomType,f,nComps)
	print roomType
	print map(lambda x: probsByTime[x]/nFolds,probsByTime)

for c in ["BR", "BH", "SH", "O", "P"]:
	getAccuracyOverTime(c, 64, 4,30)
