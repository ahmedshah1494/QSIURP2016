#!/bin/bash

folds=`seq 0 3`
declare -a classes=("BH" "SH" "BR" "O" "P")
errd="errflsGMM"
logd="logflsGMM"

for f in $folds
do
	for c in ${classes[@]}
	do
		qsub -q hp -N gridSearch.$f.$c.txt -e $errd/err.gridSearch.$f.$c.txt -o out.gridSearch.$f.$c.txt -v fld=$f,cls=$c,logf=$logd/log.gridSearch.$f.$c.txt grid.sh
		#echo SVMTrainVectors_$c"_fld$f".txt 
		python ../../libsvm-3.21/tools/grid.py ../files/folds/temp/$c/SVMTrainVectors_$c"_fld"$f.txt
	done
done
