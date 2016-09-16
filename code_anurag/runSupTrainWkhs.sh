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
mkdir -p $suplist/temp
fold=`seq 1 $fldsz`
#declare -a fold=(3)
one=1
subFolder="withSannan_64ms"
declare -a classes=('SH' 'BH' 'BR' 'O' "P")

if [ "$WorF" == "Full" ];then
    echo " Not set "
elif [ "$WorF" == "Folds" ];then
	for f in $fold
	do
	    for c in "${classes[@]}"
	    #for f in "${fold[@]}" 
	    do
	    f=$(($f-$one))
	    # echo $f
	    cd $suplist/$c/$subFolder
	    # mkdir -p temp/$c
	    posFiles=$(find -maxdepth 1 -regex '.*'$c'_p.fold+[^'$f']')
	    negFiles=$(find -maxdepth 1 -regex '.*'$c'_n.fold+[^'$f']')

	    posFileList=$suplist/temp/$c/$c'_p_'fold$f"_train"$listend
	    negFileList=$suplist/temp/$c/$c'_n_'fold$f"_train"$listend
	    
	    cat $posFiles > $posFileList
	    cat $negFiles > $negFileList

	    sed -i.bak "s|files/|../files/|" $posFileList
	    sed -i.bak "s|files/|../files/|" $negFileList

	    # echo $posFiles
	    # find -maxdepth 1 -regex '.*'$c'_n.fold+[^'$f']'
	    # echo $(find -maxdepth 1 -regex '.*'$c'.fold+[^'$f']')
	    # echo $(($f-$one))
	    # echo $c'.fold'
	    cd $currPath
	 #    supfldlist=$suplist/temp/fold$f"_"train$listend
		# # supfldlist=$suplist/$WorF/fold$f/fold$f"_"train$listend
		# tmpsupfl=temp/$trds.fold$f.gtrain.$nComp$listend.sup
		# sed "s|../|../supdata/|" $supfldlist > $tmpsupfl
		# wekfldlist=$weklist/Full/Full$listend
		# tmpwekfl=temp/$trds.fold$f.Full.gtrain.$nComp$listend.weaksup
		# sed "s|../|../weakdata/|" $wekfldlist > $tmpwekfl
		# combfl=temp/$trds.fold$f.gtrain.Full.$nComp$listend.merge
		
		# evJbnm=$trds"."$f"."$nComp
		errPt=$errd/"err."$evJbnm
		outPt=$errd/"out."$evJbnm
		logfl=$log/"log."$evJbnm

		#sed -i.bak "s|files/|../files/|" $posFileList
        #sed -i.bak "s|files/|../files/|" $negFileList
		# gmmfolder=$gmbsdr/fold$f/$gmstr
		# cd ..
		# python code_anurag/aGMMSklearn.py $posFileList $nComp train GMMs/$c/fold_$f/P/ &
		# python code_anurag/GMMSklearn.py $negFileList $nComp train GMMs/$c/fold_$f/N/ &
	 #    	echo $gmmfolder
		# qsub -q $qsz -N $evJbnm -e $errPt -o $outPt -v slist=$tmpsupfl,wlist=$tmpwekfl,mglist=$combfl,nCmp=$nComp,gmfold=$gmmfolder,logf=$logfl GMMTrainWkhs.sh
	 	qsub -q hp -N GMMsup-train_$c"_"$subFolder"_"$f"_"$nComp"_P" -e $errPt"$c.$subFolder.fold$f.$nComp.P" -o $outPt -v inFileList=$posFileList,nCmp=$nComp,outFile=GMMs/$subFolder/$c/fold_$f/P/,logf=$logfl"$c.$subFolder.fold$f.$nComp.P" GMMTrainWkhs.sh
	 	qsub -q hp -N GMMsup-train_$c'_'$subFolder'_'$f'_'$nComp'_N' -e $errPt"$c.$subFolder.fold$f.$nComp.N" -o $outPt -v inFileList=$negFileList,nCmp=$nComp,outFile=GMMs/$subFolder/$c/fold_$f/N/,logf=$logfl"$c.$subFolder.fold$f.$nComp.N" GMMTrainWkhs.sh
	 #    #echo "$evJbnm submitted"
		
	    done
	done

fi
