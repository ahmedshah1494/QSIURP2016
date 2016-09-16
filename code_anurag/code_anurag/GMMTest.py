import os
import numpy as np
from sklearn import mixture
import pickle
import sys

roomDic = {'BH': ['1185','1202','1213','2152'],
 'BR': ['C1','C2','FC1','FC2','LB2'],
 'O' : ['1004','1005','1008','1009','1018'],
 'P' : ['1192','2012','2170','CS'],
 'SH': ['1030','1190','2052','2147']}

def loadGMM(gmmFile):
	gmm = pickle.load(open(gmmFile,'rb'))
	gmm.covars_ = np.load(gmmFile+'_01.npy')
	gmm.weights_ = np.load(gmmFile+'_02.npy')
	gmm.means_ = np.load(gmmFile+'_03.npy')
	return gmm

def testFiles(filelist, ncomps, outfile, label,gmmFileDir):
	gmm_P = loadGMM(gmmFileDir+"P/"+str(ncomps)+"/sklearnGMM.pkl")
	gmm_N = loadGMM(gmmFileDir+"N/"+str(ncomps)+"/sklearnGMM.pkl")
	# gmmext_P = gmmFileDir+"P/"+str(ncomps)+"/sklearnGMM.pkl"
	# gmmext_N = gmmFileDir+"N/"+str(ncomps)+"/sklearnGMM.pkl"
	# print 'loading gmm'
	# gmm_P = pickle.load(open(gmmFile,'rb'))
	# x = np.load(gmmext_P+'_01.npy')
	# gmm_P.covars_=x
	# # print x
	# x = np.load(gmmext_P+'_02.npy')
	# gmm_P.weights_=x
	# # print x
	# x = np.load(gmmext_P+'_03.npy')
	# gmm_P.means_=x
	# print x
	# return
	print 'reading filelist:', filelist
	flist = open(filelist, 'r')
	files = flist.readlines()
	flist.close()

	dic = {"BR":[], "BH":[], "SH":[], "O":[], "P":[]}
	print roomDic
	for t in roomDic:
		print t, dic
		rooms = roomDic[t]
		for room in rooms:
			isPresent = reduce(lambda a,b: a or b, map(lambda x : room in x, files))
			if isPresent and dic[t] == []:
				dic[t].append(room)
			print room, dic
	print dic
	#for c in dic:
	#	if len(dic[c]) > 1:
	#		files = filter(lambda x : dic[c][1] not in files)
	#print dic
	'compiling data'
	alldata = None
	# files = map(lambda x: '../'+x, files)
	np.random.shuffle(files)
	out = open(outfile, 'w')
	for fl in files:
		# print "reading ", fl
		data = np.loadtxt(fl.strip(),delimiter=',')
		# data = data.reshape(1, -1)
		# alldata = alldata.getA1()
		ll_P = sum(gmm_P.score(data))
		ll_N = sum(gmm_N.score(data))
		
		out.write("%s %s %s\n" % (fl.strip(), str(ll_P), str(ll_N)))
	out.close()

if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit()
    if len(sys.argv) == 6:
    	print "HELLO"
        testFiles(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5])
    else:
        print "arg1 - in file list, arg2 - nComp, arg3 - output file, argv4-actaul label, argv5-gmm output Folder"
        sys.exit()

# testFiles('../files/folds/BR/BR_p.fold0', '../GMMs/BR/fold_0/', 1, "test_result.txt")
