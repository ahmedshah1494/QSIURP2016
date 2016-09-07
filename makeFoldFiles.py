import os
d = 'files/'
files = os.listdir(d)
dic = {'SH' : filter(lambda x : (x.split('_')[0] == 'SH') and os.path.isdir(d+x) and '_M' not in x and '_R' not in x, files),
	 'BH' : filter(lambda x : (x.split('_')[0] == 'BH') and os.path.isdir(d+x) and '_M' not in x and '_R' not in x, files),
	 'BR' : filter(lambda x : (x.split('_')[0] == 'BR') and os.path.isdir(d+x) and '_M' not in x and '_R' not in x and "AHMED" not in x, files),
	 'P' : filter(lambda x : (x.split('_')[0] == 'P') and os.path.isdir(d+x) and '_M' not in x and '_R' not in x, files),
	 'O' : filter(lambda x : (x.split('_')[0] == 'O') and os.path.isdir(d+x) and '_M' not in x and '_R' not in x, files)}

# print dic
rooms = {'BH': ['1185','1202','1213','2152'],
		 'BR': ['C1','C2','FC1','FC2','LB2'],
		 'O' : ['1004','1005','1008','1009','1018'],
		 'P' : ['1192','2012','2170','CS'],
		 'SH': ['1030','1190','2052','2147']}

def makeFoldLists(n):
	for i in range(n):
		f = open('files/folds/fold_'+str(i)+'.txt','w')
		for c in rooms:
			f.write(c+'_'+rooms[c][i] + "\n")
			if i == n-1 and len(rooms[c]) > n:
				for j in range(i+1,len(rooms[c])):
					f.write(c+'_'+rooms[c][j] + "\n")
		f.close()

def makeFolds():
	foldFiles = os.listdir('files/folds/')
	foldFiles = filter(lambda x: 'fold_' in x, foldFiles)

	for c1 in rooms:
		for i in range(len(foldFiles)):
			f_fold_p = open('files/folds/%s/%s_p.fold%d' % (c1,c1,i), 'w')
			f_fold_n = open('files/folds/%s/%s_n.fold%d' % (c1,c1,i), 'w')
			ff = foldFiles[i]
			f = open('files/folds/'+ff, 'r')
			folders = map(lambda x: x[:-1], f.readlines())
			f.close()
			for folder in folders:
				dataFiles = map(lambda x: ('files/%s_H/' % folder) + x, os.listdir('files/%s_H/' % folder))
				dataFiles += map(lambda x: ('files/%s_P/' % folder) + x, os.listdir('files/%s_P/' % folder))
				dataFiles = filter(lambda x: x.split('.')[-1] == 'feat', dataFiles)
				# print dataFiles
				for dataFile in dataFiles:
					if folder.split('_')[0] == c1:
						# label = '+1'
						f_fold_p.write(dataFile+'\n')
					else:
						# label = "-1"
						f_fold_n.write(dataFile+'\n')
# makeFoldLists(4)
makeFolds()

