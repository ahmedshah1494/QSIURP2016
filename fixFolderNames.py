import os
import shutil
d = 'files/'
files = os.listdir('files/')
# files = filter(lambda x : "BR" in x, files)
# for f in files:
# 	if os.path.isdir(d+f):
# 		dataFiles = os.listdir(d+f)
# 		wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
# 		for wav in wavs:
# 			if f not in wavs:
# 				print f
# 				split = wav.split('_')
# 				os.rename(d+f+'/'+wav, "%s%s/%s_%s_%s_%s" % (d,f,f,split[-3],split[-2],split[-1]))

types = ["BR",'LH','O','P']
dic = {"BR": ["C2","FC2","LB2","C1","FC1"],
	   "LH" : ["1190",'1202','2052','1213','1030'],
	   "P" : ['CS','1192','2170','2012'],
	   "O" : ['1009','1005','1004','1008','1018']}
recorderFolders = filter(lambda x : "DSS" in x, files)
recorderFolders.sort()
print recorderFolders
for t in dic:
	for r in dic[t]: 
		D = "files/%s_%s_R" % (t,r)
		if (not os.path.exists(D)):
			os.mkdir(D)
for j in range(len(recorderFolders)):
	folder = recorderFolders[j]
	files = os.listdir(d+folder)
	files = filter(lambda x : ".WAV" in x, files)
	for i in range(len(files)):
		rIdx = i / 20
		# if rIdx == 1:
		# 	break
		S = (i % 20) / 5
		recNum = (i % 20) % 5
		# print i, j, rIdx
		D = '%s_%s_R' % (types[j],dic[types[j]][rIdx])
		src = d+folder+'/'+files[i]
		dest = d+('%s/%s_L4_S%d_%d.wav' % (D,D,S,recNum))
		print "%s --> %s" % (src,dest)
		shutil.copy(src, dest)
		





