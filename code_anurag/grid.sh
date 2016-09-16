#!/bin/bash 
#PBS -l nodes=1:ppn=8
module load python27
module load python27-extras

cd $PBS_O_WORKDIR

#python ../../libsvm-3.21/tools/grid.py ../files/folds/temp/$class/SVMTrainVectors_$class"_fld_"$f.txt
echo ../files/folds/temp/$cls/SVMTrainVectors_$cls"_fld$fld".txt
python ../../libsvm-3.21/tools/grid.py ../files/folds/temp/$cls/SVMTrainVectors_$cls"_fld"$fld.txt > $logf
