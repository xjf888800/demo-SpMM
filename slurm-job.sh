#!/bin/bash
#SBATCH -J JOB
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=dcu:4
#SBATCH --mem=90G

echo "###### [1] SET ENVIRONMENT ######"
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module unload compiler/dtk/21.10.1
module load compiler/dtk/22.10.1
module list

echo "###### [2] COMPILE ######"
cd ${HOME}/work/home/ac3mzrt9ej/xjf/cpu_demo_SpMM/demo-SpMM/
make clean gpu=1
make gpu=1

echo "###### [3] RUN TEST: ######"
echo "###### A:random 10000x10000 spa=0.00007, B:random 50000x50000 spa=0.0001 ######"

./Csrsparse_rocsparse 0 50000 50000 0.00007 0.0001 1234

echo "###### ALL TEST FINISHED ######"

