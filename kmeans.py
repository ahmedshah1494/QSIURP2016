from sklearn import cluster
import numpy as np 
import dispy
import os

main_dir = "/Users/Ahmed/Downloads/QSIURP2016/"
dataset_dir = main_dir + 'files/DataSets/'
files_dir = main_dir + 'files/'
results_dir = main_dir + 'results/'

def learn(trainFile, K):
	with open(dataset_dir + trainFile,'r') as f:
		lines = f.readlines()
	lines = filter(lambda x : x.find("NaN") == -1 and x.find('nan') == -1, lines)
	lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)

	# kmc = cluster.MiniBatchKMeans(n_clusters=K, batch_size = 10000, init_size=len(lines), reassignment_ratio=0.1, verbose=1)
	kmc = cluster.KMeans(n_clusters=K, n_jobs= -1)
	kmc.fit(lines)

	return kmc

def quantizeFile((filename, kmc)):
	main_dir = "/Users/Ahmed/Downloads/QSIURP2016/"
	files_dir = main_dir + 'files/'
	from sklearn import cluster
	
	with open(files_dir+filename, 'r') as f_feat:
			lines = f_feat.readlines()

	lines = filter(lambda x : x.find("NaN") == -1 and x.find('nan') == -1, lines)
	lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)

	res = kmc.predict(lines)

	dic = []
	dic = [0] * (kmc.get_params()['n_clusters'])

	for c in res:
		dic[c] += 1

	return dic

def quantizeFolder(folders, kmc, diff=False, normalize=False, C_0=False):
	# print folders
	if normalize:
		feat_ext = "Nfeat"
		train_ext = "Ntrain"
		quant_ext = ".Nquant"
	elif diff:
		feat_ext = "Dfeat"
		train_ext = "Dtrain"
		quant_ext = ".Dquant"
	elif C_0:
		feat_ext = "Cfeat"
		train_ext = "Ctrain"
		quant_ext = ".Cquant"
	else:
		feat_ext = "feat"
		train_ext = "train"
		quant_ext = ".quant"

	node1 = dispy.NodeAllocate("0.0.0.0",port=9998)
	# node 2 is a remote machine
	node2 = dispy.NodeAllocate('10.33.49.230', port=9993)
	node3 = dispy.NodeAllocate('10.33.49.17', port=9991)
	node4 = dispy.NodeAllocate('10.33.49.32', port=9992)
	nodeList = [node1]

	testCluster = dispy.JobCluster(quantizeFile, nodeList)
	files = map(lambda x : map(lambda x2: x+'/'+x2, filter(lambda x3 : x3.split('.')[-1] == feat_ext, os.listdir(files_dir + x))), folders)
	files = reduce(lambda x,y: x + y, files)
	# print files
	jobs = map(lambda x : testCluster.submit((x, kmc)), files)
	results = map(lambda j : j(), jobs)

	for i in range(len(results)):
		# print files[i], results[i]
		if results[i] == None:
			continue
		with open(files_dir+files[i]+quant_ext,'w') as f:
			line = reduce(lambda x,y : x + ',' + y, map(str,results[i]))
			f.write(line + '\n')
	# print results

kmc = learn("universal.Ntrain",1024)
quantizeFolder(filter(lambda x : os.path.isdir(files_dir + x), os.listdir(files_dir)), kmc, normalize=True)
