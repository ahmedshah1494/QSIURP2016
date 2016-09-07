import sys
import os
import numpy
import time

from sklearn import mixture
from sklearn.externals import joblib


mxVec = 1500000 # max number of vectors for GMM training -- set to -1 to use all
skgmm = 'sklearnGMM'
covType = 'diag'
random_state = None
tl = 0.0001
mincovr = 0.0001
niter = 150
ninit = 1
paramu = 'wmc'
iparam = 'wmc'
cov_tol = 0.0001

#need to to put all of these parameters properly -- may be using yaml
#currently using indices to train on no delta or any such thing
stid=0
edid=60 #use last+1 as in numpy notation
normft=0 # if set to 1 -- 0 mean and unit variance normalization before training GMM.. these are also stored in the same directory
libformat=0 # whether to store in libsvm format or not

#concatenation is slow-need to fix it
#preallocate best perhaps
#list also option but memory concerns
def trainGMM(fileList,nComp,outFld):
    ''' Train GMM using the list of files
    Parameters
    ----------
    fileList - list of files containing vectors
    nComp - Number of Gaussians
    outFld - Output Folder To store GMM

    Output
    -------
    saves GMM in .pkl Form as well as means, weights and covarainces in ascii format
    '''

    
    fx = open(fileList)
    fllist = fx.readlines()
    print 'Loading Data...'
    tt=5000
    for fl in fllist:
        
        cdata = numpy.loadtxt(fl.strip(),delimiter=',')
        cdata = cdata[:,stid:edid]
        if not 'alldata' in locals():
            alldata = cdata
        else:
            alldata = numpy.concatenate((alldata,cdata),axis=0)
            if alldata.shape[0] > tt:
                print tt
                tt=tt+5000
    print 'Number of Vectors = ' + str(alldata.shape[0])

    gmmSvPath = os.path.join(outFld,str(nComp))
    if not os.path.isdir(gmmSvPath):
        os.makedirs(gmmSvPath)

    if normft == 1:
        print '0 mean and unit variance normalization'
        aldmn = alldata.mean(axis=0)
        aldst = alldata.std(axis=0)

        alldata = alldata - aldmn
        alldata = alldata/aldst
        numpy.savetxt(os.path.join(gmmSvPath,'0meanNorm.txt'),aldmn,delimiter=' ')
        numpy.savetxt(os.path.join(gmmSvPath,'1stdNorm.txt'),aldst,delimiter=' ')
        
    if mxVec == -1 or mxVec > alldata.shape[0]:
        mxVecTk = alldata.shape[0]
    else:
        mxVecTk = mxVec

    print 'Training On ' + str(mxVecTk) + ' Vectors with ' + str(nComp) + ' components' 

    numpy.random.shuffle(alldata) #shuffle
    alldata = alldata[0:mxVecTk,:]

    gmm = mixture.GMM(n_components=nComp,covariance_type=covType,random_state=None,thresh=None,tol=tl,min_covar=mincovr,n_iter=niter,n_init=ninit,params=paramu,init_params=iparam,verbose=1)
    #gmm = mixture.GMM(n_components=nComp,covariance_type=covType,random_state=None,thresh=None,tol=tl,min_covar=mincovr,n_iter=niter,n_init=ninit,params=paramu,init_params=iparam) #no verbose in version before 16.0
    gmm.fit(alldata)
    
    joblib.dump(gmm,os.path.join(gmmSvPath,skgmm+'.pkl'))
    
    wts = numpy.reshape(gmm.weights_,(1,-1))
    
    allmns = gmm.means_
    allsgm = gmm.covars_
    
    numpy.savetxt(os.path.join(gmmSvPath,'weights.txt'),wts,delimiter=' ')
    numpy.savetxt(os.path.join(gmmSvPath,'means.txt'),allmns,delimiter=' ')
    if covType != 'full':
        numpy.savetxt(os.path.join(gmmSvPath,'covariances.txt'),allsgm,delimiter=' ')
    else:
        allsgm = numpy.reshape(allsgm,(allsgm.shape[0]*allsgm.shape[1],allsgm.shape[2]))
        numpy.savetxt(os.path.join(gmmSvPath,'covariances.txt'),allsgm,delimiter=' ')


def doBoW(inList,nComp,gmfold,outList):
    ''' bag of words representation for list of files
    ----------
    inList - list of files containing vectors
    nComp - Number of Gaussians
    gmfold - head folder where all gmms are stored
    outList - Output Folder To store GMM

    Output
    -------
    saves bag of words representation in libsvm format at outList locations
    '''

    gmpth = os.path.join(gmfold,str(nComp),skgmm+'.pkl')
    gmm = joblib.load(gmpth)
    

    ifx = open(inList)
    ifllist = ifx.readlines()

    ofx = open(outList)
    ofllist = ofx.readlines()
    
    if len(ifllist) != len(ofllist):
        raise ValueError('Length of Input and Output list do not match')

    if normft == 1:
        print '0 mean and unit variance normalization'
        aldmn = numpy.loadtxt(os.path.join(gmfold,str(nComp),'0meanNorm.txt'))
        aldst = numpy.loadtxt(os.path.join(gmfold,str(nComp),'1stdNorm.txt'))


    for i,fl in enumerate(ifllist):
        if not (os.path.isfile(fl.strip())):
            print fl.strip() + ' Not Found'
        else:
            cdata = numpy.loadtxt(fl.strip())
            cdata = cdata[:,stid:edid]
            if normft == 1:
                cdata = cdata - aldmn
                cdata = cdata/aldst

            postrs = gmm.predict_proba(cdata)
            hist = numpy.sum(postrs,axis=0)
            histfeat = hist/float(cdata.shape[0])
            histfeat = histfeat.reshape(1,histfeat.shape[0])

            if numpy.any(numpy.isnan(histfeat)) or numpy.any(numpy.isinf(histfeat)):
                print 'Escaping because of NaN or Inf '+ fl
            else:
                if not os.path.isdir(os.path.dirname(ofllist[i].strip())):
                    os.makedirs(os.path.dirname(ofllist[i].strip()))
                    
                if libformat:    
                    with open(ofllist[i].strip(),'w') as f:    
                        f.write(" ".join(["{}:{}".format(i,histfeat[0][i]) for i in range(histfeat.shape[1]) if histfeat[0][i] != 0]))
                else:
                    numpy.savetxt(ofllist[i].strip(),histfeat)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "arg1 - in file list, arg2 - nComp, arg3 - train/bow, argv4-gmm output Folder, arg5 - bow filelist(only for arg3 = bow)"
        sys.exit()
    if sys.argv[3] == 'bow' and len(sys.argv) == 5:
        print "arg1 - in file list, arg2 - nComp, arg3 - train/bow, argv4-gmm output Folder, arg5 - bow filelist(only for arg3 = bow)"
        sys.exit()
    if sys.argv[3] == 'train':
        trainGMM(sys.argv[1],int(sys.argv[2]),sys.argv[4])
    elif sys.argv[3] == 'bow':
        doBoW(sys.argv[1],int(sys.argv[2]),sys.argv[4],sys.argv[5])
    else:
        print "arg1 - in file list, arg2 - nComp, arg3 - train/bow, argv4-gmm output Folder, arg5 - bow filelist(only for arg3 = bow)"
        sys.exit()
