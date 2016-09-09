#!/bin/bash

declare -a classes=('SH' 'BH' 'BR' 'O' "P")
fldz=$(seq 1 1)
nComps=1
one=1
currDir=$(pwd)
errd="errflsGMM"
log="logflsGMM"

for c in "${classes[@]}"
do
	for f in $fldz
	do
		f=$(($f-$one))

		gmmDir="../GMMs/$c/fold_$f/"
		resultDir="results/$c/fold_$f/$nComps/"
		mkdir -p $resultDir

		posFile=../files/folds/$c/$c"_p.fold"$f
		negFile=../files/folds/$c/$c"_n.fold"$f

		errPt=$errd/"err."$evJbnm
		outPt=$errd/"out."$evJbnm
		logfl=$log/"log."$evJbnm
		tempDir="../files/folds/temp/$c/"
		posFileList=$tempDir$c"_p_fold"$f"_test.list"
		negFileList=$tempDir$c"_n_fold"$f"_test.list"
		sed "s|files/|../files/|" $posFile > $posFileList
        sed "s|files/|../files/|" $negFile > $negFileList

        # echo $tempDir$c"_p_fold"$f"_test.list"
        python GMMTest.py $posFileList $nComps $resultDir"P.txt" 1 $gmmDir
		# qsub -q hp -N GMM_Testing_$c_fold$f_P $errPt"test.$c.fld$f.$nComps.P" -o $outPt -v fileList=$posFile,nCmps=$nComps,$outFile=$resultDir"P.txt",$label=1,$gmmFileDir=$gmmDir"P",logf=$logfl 
		# qsub -q hp -N GMM_Testing_$c_fold$f_P $errPt"test.$c.fld$f.$nComps.N" -o $outPt -v fileList=$negFile,nCmps=$nComps,$outFile=$resultDir"N.txt",$label=1,$gmmFileDir=$gmmDir"N",logf=$logfl 
	done
done