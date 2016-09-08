#!/bin/bash

#PBS -l nodes=1:ppn=8
module load python27
module load python27-extras

cd $PBS_O_WORKDIR

#echo $slist

python GMMSklearn.py $inFileList $nCmp train $outFile > $logf
