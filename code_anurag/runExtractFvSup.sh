#!/bin/bash

#this uses the qsub to submit jobs
#Remember - 

nComp=64
trds=dcase_acoustic_scenes
fldsz=4
SorW=W # supervised or weakly supervised .. for weakly supervised everything is done for all recordings for all GMMs accroding to Folds
WhorSe=Seg # Whole or Seg ##
segl=30 #segment length 
tocomp='S' #F, S or FS

WorF=Folds 
qsz=long

feat1=fvmfcc
feat2=supvmfcc
infeat=mfcc
gmstr=sklearn
listend=.mfcc.list
suplist=../supdata/lists/$trds
weklist=../weakdata/lists/$trds
gmbsdr=../GMMs/$trds/$WorF


errd="errflsFvSup"
log="logflsFvSup"

mkdir -p $errd 
mkdir -p $log
mkdir -p temp

#fold=`seq 1 $fldsz`
declare -a fold=(4)

if [ "$WorF" == "Full" ];then
    echo "Not Set"
    
elif [ "$WorF" == "Folds" ];then
    
    #for f in $fold
    for f in "${fold[@]}"
    do
	
	#input will be whole list again -- need to extract feats for all
	if [ "$SorW" == "S" ];then
	    supfldlist=$suplist/Full/Full$listend
	    ifl=`mktemp --tmpdir=temp`
	    rm -f $ifl
	    infl=$ifl.$evJbnm
	    sed "s|../|../supdata/|" $supfldlist > $infl
	
	elif [ "$SorW" == "W" ];then
	    wekfldlist=$weklist/Full/Full$listend
	    ifl=`mktemp --tmpdir=temp`
	    rm -f $ifl
	    infl=$ifl.$evJbnm
	    sed "s|../|../weakdata/|" $wekfldlist > $infl
	fi
	
	gmmfolder=$gmbsdr/fold$f/$gmstr
	if [ "$tocomp" == "F" ];then
	    evJbnm=$feat1.$WhorSe.$SorW.$WorF.$trds.$nComp.$f
	    errPt=$errd/"err."$evJbnm
	    outPt=$errd/"out."$evJbnm
	    logfl=$log/"log."$evJbnm
	    
	    
	    tfl=`mktemp --tmpdir=temp`
	    rm -f $tfl
	    tfl=$tfl.$evJbnm
	    sed "s|$infeat|$feat1|g" $infl > $tfl
	    
	    if [ "$WhorSe" == "Whole" ];then
		sed -i.bak "s|$trds/|$trds/Folds/fold$f/$gmstr/$WhorSe/$nComp/|" $tfl
		#qsub -q $qsz -N $evJbnm -e $errPt -o $outPt -v infllist=$infl,nCmp=$nComp,gmfold=$gmmfolder,tocmp=$tocomp,outl1=$tfl,logf=$logfl ExtractFvSupWhole.sh
	    elif [ "$WhorSe" == "Seg" ];then
		sed -i.bak "s|$trds/|$trds/Folds/fold$f/$gmstr/$WhorSe$segl/$nComp/|" $tfl
		qsub -q $qsz -N $evJbnm -e $errPt -o $outPt -v infllist=$infl,nCmp=$nComp,gmfold=$gmmfolder,tocmp=$tocomp,outl1=$tfl,seglen=$segl,logf=$logfl ExtractFvSupSeg.sh
	    else
		echo "Whole or Seg"
	    fi
	    
	elif [ "$tocomp" == "S" ];then
	    
	    evJbnm=$feat2.$WhorSe.$SorW.$WorF.$trds.$nComp.$f
	    errPt=$errd/"err."$evJbnm
	    outPt=$errd/"out."$evJbnm
	    logfl=$log/"log."$evJbnm
	    
	    tfl=`mktemp --tmpdir=temp`
	    rm -f $tfl
	    tfl=$tfl.$evJbnm
	    sed "s|$infeat|$feat2|g" $infl > $tfl
	    
	    if [ "$WhorSe" == "Whole" ];then
		echo Whole
		sed -i.bak "s|$trds/|$trds/Folds/fold$f/$gmstr/$WhorSe/$nComp/|" $tfl
		qsub -q $qsz -N $evJbnm -e $errPt -o $outPt -v infllist=$infl,nCmp=$nComp,gmfold=$gmmfolder,tocmp=$tocomp,outl1=$tfl,logf=$logfl ExtractFvSupWhole.sh
	    elif [ "$WhorSe" == "Seg" ];then
		echo Seg
		sed -i.bak "s|$trds/|$trds/Folds/fold$f/$gmstr/$WhorSe$segl/$nComp/|" $tfl
		qsub -q $qsz -N $evJbnm -e $errPt -o $outPt -v infllist=$infl,nCmp=$nComp,gmfold=$gmmfolder,tocmp=$tocomp,outl1=$tfl,seglen=$segl,logf=$logfl ExtractFvSupSeg.sh
	    else
		echo "Whole or Seg"
	    fi


	elif [ "$tocomp" == "FS" ];then
	    echo "Not set"
	else
	    echo "tocomp can be F, S or FS "
	    exit 1
	fi

    done

else
    echo "Full or Folds"
    #exit 1
fi





