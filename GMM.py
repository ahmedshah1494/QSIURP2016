import numpy as np
from sklearn import mixture

def printMat(m):
	for l1 in m:
		print l1

d = 'files/DataSets/'
def learn(mixtureSize, trainFiles):
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
	for ts in train_set:
		gmm = mixture.GMM(n_components=mixtureSize,covariance_type='full',min_covar=1.0,verbose=1)
		gmm.fit(ts)
		GMMs.append(gmm)

	return GMMs

def test(testFiles, GMMs):
	test_set = []
	test_labels = []
	f_res = open("results/%dG_%s.txt" % (GMMs[0].get_params()["n_components"], filter(lambda x : x.isupper() or x =='_', reduce(lambda y,z : y + '_' + z, testFiles))), 'w')
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


for i in range(0,11,2):
	GMMs = learn((2**i), [d+"LectureHall.train", d+"Bathroom.train"])
	test([d+"LectureHall.test", d+"Bathroom.test"], GMMs)
