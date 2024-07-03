#!/bin/bash

#SBATCH -e slurm-%j.err
#SBATCH -D /home/savancha/GNN/dgl-mlperf/applications/IGBH
#SBATCH -J gnn_conv
#SBATCH --get-user-env
#SBATCH --partition=emr
#SBATCH --nodes=16
#SBATCH --nodelist=emr[01-16]

source /opt/intel/oneapi/mpi/2021.12/env/vars.sh

# Setup MPI Env
export FI_PROVIDER=psm3
export PSM3_IDENTIFY=1
export PSM3_ALLOW_ROUTERS=1
export PSM3_RDMA=1
export PSM3_RV_MR_CACHE_SIZE=8192
export FI_PROVIDER_PATH=/usr/lib64/libfabric
export I_MPI_OFI_PROVIDER=psm3
export I_MPI_MULTIRAIL=1
export PSM3_MULTIRAIL=2

python --version

bash ./run_multi.sh $1 $2 $3 $4 $5 $6
