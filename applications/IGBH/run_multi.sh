#!/bin/bash

if [ "$#" != "6" ]; then
  echo "Usage: run_multi.sh <path-to-partitioned-dataset> <number-of-runs> <number-of-partitions> <number of procs-per-partition> <learning-rate>"
  exit
fi

export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
path=$1
r=1
while [ $r -le $2 ]
do
  export DIR=.
  bash clean.sh

  run_dist.sh -n $3 -ppn $4 \
	python -u -W ignore dist_train_rgat.py \
	--path $path --learning_rate $5 \
	--tpp_impl --use_bf16 \
        --n_classes 2983  --batch_size 1024 \
	--hidden_channels 512 --ielsqsize 1 \
	--val_batch_size 2048 --val_fraction 0.025 \
	--n_epochs 2 --model_save \
	--target_acc $6

  r=$(( $r + 1 ))
done
