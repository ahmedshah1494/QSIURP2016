import numpy as np
import os
import matplotlib.pyplot as plt
subfolder = "SV"
def getAccuracyOverTime(roomType,nComps,nFolds,interval,resultsFile, svm=False):
	# interval in seconds
	probs_PByTime = {}

	for f in range(nFolds):
		path_P="results/%s/fold_%d/%s/P/results.txt" % (subfolder,f,roomType)
		path_N="results/%s/fold_%d/%s/N/results.txt" % (subfolder,f,roomType)
		# if not os.path.exists("results/%s/%s/analysis/fold_%d/%d/" % (subfolder,f,roomType)):
		# 	os.makedirs("results/%s/%s/analysis/fold_%d/%d/"  % (subfolder,f,roomType))
		# resultPath = "results/%s/%s/analysis/fold_%d/%d/result.txt" % (subfolder,f,roomType))
		fl_P = open(path_P, 'r')
		fl_N = open(path_N, 'r')
		# resultFl = open(resultPath, 'w')
		lines_P_P = fl_P.readlines_P()
		lines_P_N = fl_N.readlines_P()
		fl_P.close()
		fl_N.close()

		if svm:
			labels_P = map(lambda x : (map(int,x.split()[1:]))[0], lines_P_P)
			labels_P = map(lambda x: 0 if x == -1 else 1, labels_P)
			labels_N = map(lambda x : (map(int,x.split()[1:]))[0], lines_P_N)
			labels_N = map(lambda x: 0 if x == -1 else 1, labels_N)
		else:
			lines_P_P = map(lambda x : (map(float,x.split()[1:])), lines_P_P)
			labels_P = map(lambda x: 1 ^ np.argmax(x),lines_P_P)

			lines_P_N = map(lambda x : (map(float,x.split()[1:])), lines_P_N)
			labels_N = map(lambda x: 1 ^ np.argmax(x),lines_P_N)


		lines_PPerInterval = interval/3
		prevProb_P = 1.0
		prevProb_N = 1.0
		for i in range(lines_PPerInterval,len(lines_P_P),lines_PPerInterval):
			prob_P = ((sum(labels_P[:i])/float(i)));
			prob_N = (sum(labels_N[:i])/float(i));
			# resultFl.write("%d %f %f\n" % (i*3, prob_P, prob_N))
			if probs_PByTime.get(i*3) == None:
				probs_PByTime[i*3] = []
			probs_PByTime[i*3].append(np.array([prob_P, prob_N]))

			# prob = sum(map(np.array, lines_P_P[:i]))
			# prob = prob[0]/sum(prob)
			# if probs_PByTime.get(i*3) == None:
			# 	probs_PByTime[i*3] = []
			# probs_PByTime[i*3].append(prob)
			# resultFl.write("%d %f\n" % (i*3, prob))

			# total = sum(map(np.array, lines_P[:i]))
			# if prevProb == 1:
			# 	prob = total[1]
			# else:
			# 	prob = (total[1]/prevProb)
			# prevProb = total[1]
			# print prob
			# # print total, sum(map(sum,lines_P[:i]))

		# path_N="results/%s/fold_%d/%d/N.txt" % (roomType,f,nComps)
	# print roomType
	# print map(lambda x: probs_PByTime[x]/nFolds,probs_PByTime)
	# resultFl.close()
	# print probs_PByTime
	probs_PByTime2 = probs_PByTime.copy()
	for c in probs_PByTime2:
		if len(probs_PByTime[c]) < 4:
			probs_PByTime.pop(c,None)
		else:
			probs_PByTime[c] = sum(probs_PByTime[c])/len(probs_PByTime[c])

	plt.plot(sorted(probs_PByTime),map(lambda x: (probs_PByTime[x][0]) * 100, sorted(probs_PByTime)))
	plt.plot(sorted(probs_PByTime),map(lambda x: (probs_PByTime[x][1]) * 100, sorted(probs_PByTime)))
	plt.ylim([0,100])
	plt.legend(['Percentage of True Positives', "Percentage of False Positives"], loc=0)
	plt.savefig('results/%s/analysis/%s.png' % (subfolder, roomType))
	plt.clf()
	for c in sorted(probs_PByTime):
		resultsFile.write("%s %d 		%s\n" % (roomType, c, str(probs_PByTime[c])))

def makeDETCurve(resFolder, duration, divs):
	finalFA = np.array([0.0]*divs)
	finalMD = np.array([0.0]*divs)
	for i in range(4):
		f = open(resFolder+"/fold_%d/64/P.txt" % i, 'r')
		lines_P = f.readlines()
		f.close()

		f = open(resFolder+"/fold_%d/64/N.txt" % i, 'r')
		lines_N = f.readlines()
		f.close()

		nlines = duration/3
		lines_P = lines_P[:nlines]
		lines_P = map(lambda x: map(float,x.split()[1:]), lines_P)
		probs_P = map(lambda x: 100000*(x[0] / sum(x)), lines_P)
		
		lines_N = lines_N[:nlines]
		lines_N = map(lambda x: map(float,x.split()[1:]), lines_N)
		probs_N = map(lambda x: 100000*(x[0] / sum(x)), lines_N)

		allProbs = probs_P + probs_N
		thresh = range(int(min(allProbs)), int(max(allProbs)), (int(max(allProbs) - min(allProbs))/divs)) 
		
		MD = []
		FA = []
		for th in thresh:
			labels = []
			for p in probs_P:
				if p > th:
					labels.append(0)
				else:
					labels.append(1)
			MD.append(sum(labels)/float(len(labels)))

			labels = []
			for p in probs_N:
				if p > th:
					labels.append(1)
				else:
					labels.append(0)
			FA.append(sum(labels)/float(len(labels)))
		MD = np.array(MD[:divs])
		FA = np.array(FA[:divs])

		finalMD += MD
		finalFA += FA

		# print th, labels
	finalMD = map(lambda x: x/4, finalMD)
	finalFA = map(lambda x: x/4, finalFA)

	eqLinePts = map(lambda x: x/float(divs), range(divs))
	myPoint = map(lambda x: [finalFA[x], finalMD[x]], range(divs))
	diffs = map(lambda x : np.sqrt((myPoint[x][0] - eqLinePts[x])**2 - (myPoint[x][1] - eqLinePts[x])**2), range(divs))
	# print eqLinePts	
	# diffs = map(lambda x: (finalFA[x] - x)**2 + (finalMD[x] - eqLinePts[x]))
	# # print finalMD
	diffs = map(abs, np.array(finalMD) - np.array(finalFA))
	print diffs
	minDiffIdx = np.argmin(diffs)
	# minDiffIdx = np.argwhere(np.isclose(finalMD, eqLinePts)).reshape(-1)
	print minDiffIdx
	print resFolder, finalMD[minDiffIdx]

	thresh = map(lambda x: x/100000.0, thresh)
	plt.plot(finalFA,finalMD)
	# print finalFA
	# print finalMD
	
	
for c in ["BR", "BH", "SH", "O", "P"]:
	makeDETCurve("results/withSannan_64ms/testOnSingleRoom/"+c, 600, 300)
eqLinePts = map(lambda x: x/1000.0, range(1000))
plt.plot(eqLinePts,eqLinePts)
plt.ylabel("False Negative %")
plt.xlabel("False Positive %")
plt.legend(['Bathrooms', "Big Lecture Halls", 'Classrooms', 'Offices', 'Pantries'], loc=0)
	# plt.plot(finalFA,thresh)
# plt.show()
plt.savefig('results/withSannan_64ms/testOnSingleRoom/analysis/DETcurve.png')
# f = open('results/%s/analysis/results.txt' % (subfolder),'w')
# for c in ["BR", "BH", "SH", "O", "P"]:
# 	getAccuracyOverTime(c, 64, 4,30,f,svm=True)

