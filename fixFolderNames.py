import os
d = 'files/'
files = os.listdir('files/')
files = filter(lambda x : "BR" in x, files)
for f in files:
	if os.path.isdir(d+f):
		dataFiles = os.listdir(d+f)
		wavs = filter(lambda x : x.split('.')[-1] == "wav",dataFiles)
		for wav in wavs:
			if f not in wavs:
				print f
				split = wav.split('_')
				os.rename(d+f+'/'+wav, "%s%s/%s_%s_%s_%s" % (d,f,f,split[-3],split[-2],split[-1]))

