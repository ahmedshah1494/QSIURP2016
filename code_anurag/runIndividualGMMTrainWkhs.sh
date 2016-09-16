#!/bin/bash

#this uses the qsub to submit jobs
#Remember - For Each Fold - the GMM is trained on rest of folds. So GMM stored in fold1  is trained on all folds except fold1.

nComp=64
trds=ESC-10
fldsz=4
qsz=hplong
gmstr=sklearn
WorF=Folds # Full/Folds for supdata -- for weakdata it is always Full

listend=.mfcc.list
currPath="$(pwd)"
suplist=$currPath/../files/folds
# suplist=../supdata/lists/$trds
# weklist=../weakdata/lists/$trds

gmbsdr=../GMMs/$trds/$WorF


errd="errflsGMM"
log="logflsGMM"

mkdir -p $errd 
mkdir -p $log
#mkdir -p $suplist/temp
fold=`seq 1 $fldsz`
#declare -a fold=(3)
one=1
subFolder="withSannan_64ms"
declare -a classes=('SH' 'BH' 'BR' 'O' "P")

#if [ "$WorF" == "Full" ];then
 #   echo " Not set "
#elif [ "$WorF" == "Folds" ];then
	#for c in "${classes[@]}"
	#do
	    #for f in $fold
	    #for f in "${fold[@]}" 
	    #do
	    f=$(($f-$one))
	    # echo $f
	    cd $suplist/../
	    # mkdir -p temp/$c
	    featFiles=$(find -maxdepth 2 -regex '.*/*.feat64ms')
	    for fl in $featFiles:
	    do
		trainFile=../files/$fl
	    	idxSlsh=$(expr index "${fl:3}" /)
		trainFileName=${fl:$idxSlsh}
		trainFileName=${trainFileName:3}
		
	    #negFiles=$(find -maxdepth 3 -regex '.*.feat64ms')
	    #featFileList=$suplist/temp/"featFiles_train.txt"
	    #negFileList=$suplist/temp/$c/$c'_n_'fold$f"_train"$listend
	    
	    #echo $featFiles > $featFileList
	    #cat $negFiles > $negFileList
	    #sed -i.bak "s|files/|../files/|" $featFileList
	    #sed -i.bak "s|files/|../files/|" $negFileList
	    cd $currPath
		evJbnm=$trds"."$f"."$nComp
		errPt=$errd/"err."
		outPt=$errd/"out."
		logfl=$log/"log."
		
	
		qsub -q hp -N GMM-train_$trainFileName -e $errPt"train_individual" -o $outPt -v inFileList=$trainFile,nCmp=$nComp,outFile=GMMs/individualGMMs/$trainFileName,logf=$logfl"train_individual" GMMTrainWkhs.sh
	    done	 	
#qsub -q hp -N GMM-train_$c'_'$subFolder'_'$f'_'$nComp'_N' -e $errPt"$c.$subFolder.fold$f.$nComp.N" -o $outPt -v inFileList=$negFileList,nCmp=$nComp,outFile=GMMs/$subFolder/$c/fold_$f/N/,logf=$logfl"$c.$subFolder.fold$f.$nComp.N" GMMTrainWkhs.sh
	 #    #echo "$evJbnm submitted"
		
	 #   done
	#done

#fi
