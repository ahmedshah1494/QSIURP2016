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

svDir=$currPath/GMMs/SVs


errd="errflsGMM"
log="logflsGMM"

mkdir -p $errd 
mkdir -p $log
mkdir -p $suplist/temp
fold=`seq 1 $fldsz`
#declare -a fold=(3)
one=1
subFolder="64ms"
declare -a classes=('BH' 'SH' 'O' 'P' 'BR')
makeTrainFile=1
if [ "$WorF" == "Full" ];then
    echo " Not set "
elif [ "$WorF" == "Folds" ];then
	for c in "${classes[@]}"
	do
	    for f in $fold
	    #for f in "${fold[@]}" 
	    do
	    f=$(($f-$one))
	    echo $f $c
	    #cd $suplist/$c/$subFolder
	    # mkdir -p temp/$c
	    #posFiles=$(find -maxdepth 1 -regex '.*'$c'_p.fold+[^'$f']')
	    #negFiles=$(find -maxdepth 1 -regex '.*'$c'_n.fold+[^'$f']')
	    trainingFiles=""
	    for f1 in $fold
	
	    	do
		#echo $fl
		f1=$(($f1-$one))
		if [ $f -eq $f1 ]; then
		   continue
		fi
		if [ $makeTrainFile -eq 0 ]; then
		   continue
		fi

	    	cd $svDir/fold_$f1/rooms
	   
	    #echo $f
	    
	   	 posFileList=$suplist/temp/$c/$c'_p_'fold$f"_train.svm.temp"
            	negFileList=$suplist/temp/$c/$c'_n_'fold$f"_train.svm.temp"
	    
           	 posFiles=$(find . -type f -name "$c*.feat64ms")
	    #echo $posFiles
		echo $posFIles > $suplist/temp/posSVMfiles.list
	    	echo $posFiles | tr " " "\n" > $posFileList
	    #echo $f
	    #cat $posFileList
	    	python $currPath/toSVMFormat.py $posFileList 1 , .
	    #echo $f
	    	posFiles=$(find . -type f -name "$c*.feat64ms.libsvm")
		#cat $posFiles
            #echo hello: $posFiles
	    	negFiles=$(find . -type f ! -name "*$c*")
	    echo $negFIles > $suplist/temp/negSVMfiles.list
	    	echo $negFiles | tr " " "\n" > $negFileList
            	python $currPath/toSVMFormat.py $negFileList -1 , .
            	negFiles=$(find . -type f ! -name "*$c*" | grep .libsvm)
	    #echo $negFiles
	    #echo hello$posFile
		trainFiles=$posFiles" "$negFiles
		echo $trainFiles | tr " " "\n" > $suplist/temp/$c/.temp 
		sed -i.bak "s|./|$svDir/fold_$f1/rooms/|" $suplist/temp/$c/.temp
		trainFiles=$(cat $suplist/temp/$c/.temp | tr "\n" " ")
		#cat $trainFiles > $suplist/temp/$c/SVMTrainVectors_$c"_fld$f".txt
		
	    	posFileList=$suplist/temp/$c/$c'_p_'fold$f"_train.svm.temp"
	    	negFileList=$suplist/temp/$c/$c'_n_'fold$f"_train.svm.temp"
	    
	    	#echo $posFiles | tr " " "\n" > $posFileList
	    	#echo $negFiles | tr " " "\n" > $negFileList
	    	#sed -i.bak "s|./|$svDir/fold_$f1/rooms/|" $posFileList
	    	#sed -i.bak "s|./|$svDir/fold_$f1/rooms/|" $negFileList
		#$trainingFiles+=($(cat $posFileList | tr "\n" " "))
		trainingFiles=$trainingFiles" "$trainFiles
		#echo $trainingFiles
		#trainingFiles=("${trainingFiles[@]}" "$negFileList")
		#break
	    	done
	    #echo $trainingFiles
	    cat $trainingFiles > $suplist/temp/$c/SVMTrainVectors_$c"_fld$f".txt
	    #sed -i.bak "s|./|$svDir/fold_$f1/rooms/|" $suplist/temp/$c/SVMTrainVectors_$c"_fld$f".txt
	    #cat  > $suplist/temp/$c/SVMTrainFile.txt
	    #cat $(cat $suplist/temp/$c/SVMTrainFile.txt | tr "\n" " ") > $suplist/temp/$c/SVMTrainVectors_$c"_fld$f".txt
	    
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
	 	#qsub -q hp -N SVM-Train_$c"_fld"$f -e errPt $errPt"$c.fold$f.P" -o $outPt -v inFileList="$suplist/temp/$c/SVMTrainFile.txt",svmDir=SVMs/fold_$f/$c/,logf=$logfl"fld$f.$c" SVMTrainWkhs.sh
		#qsub -q hp -N SVM-train_$c"_"$f"_"$nComp -e $errPt"svmTrain.$c.fold$f" -o $outPt -v inFileList="$suplist/temp/$c/SVMTrainFile.txt",svmDir=SVMs/$fold_$f/$c/,logf=$logfl"svmTrain.$c.fold$f" SVMTrainWkhs.sh		
			 	
	 #    #echo "$evJbnm submitted"
		#echo "$suplist/temp/$c/SVMTrainVectors"_$c"_fld$f.txt"
		qsub -q long -N SVM-Train_$c"_fld"$f -e $errPt"$c.fold$f.libsvm" -v inFile="$suplist/temp/$c/SVMTrainVectors"_$c"_fld$f.txt",logf=$logfl"libsvmTrain.$c.fold$f" libsvmTrainWkhs.sh
		#python ~/libsvm-3.21/tools/grid.py -log2c -1,2,1 -log2g 1,1,1 -t 0 $suplist/temp/$c/SVMTrainVectors.txt > grid.dump
		#declare -a gridResult=($(sed -n '$p' grid.dump))
		#C=${gridResult[0]}
		#~/libsvm-3.21/svm-train -c $C -v 5 $suplist/temp/$c/SVMTrainVectors.txt
		#../../svm-scale -l -1 -u 1 -s range1 $inFile > $inFile.scale
		#python ../../libsvm-3.21/tools/grid.py $inFile.scale > $inFile.grid.dump
		#declare -a gridResult=($(sed -n '$p' grid.dump))
		#C=${gridResult[0]}
		#G=${gridResult[1]}
	#echo ../../libsvm-3.21/svm-train -c $C $inFile > $logf
		#../../libsvm-3.21/svm-train -b 1 -c $C -g $G $inFile > $logf
	    #break	
	    done
	#break
	done

fi
