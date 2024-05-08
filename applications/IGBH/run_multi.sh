#!/bin/bash

if [ "$#" != "4" ]; then
  echo "Usage: run_multi.sh <path-to-partitioned-dataset> <number-of-runs> <number-of-partitions> <learning-rate>"
  exit
fi

export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
path=$1
r=1
while [ $r -le $2 ]
do
  export DIR=.
  bash clean.sh

  run_dist.sh -n $3 -ppn 2 -f ./hostfile \
	python -u -W ignore dist_train_rgat.py \
	--path $path --learning_rate $4 \
	--opt_mlp --use_bf16 \
        --n_classes 2983  --batch_size 1024 \
	--hidden_channels 512 \
	--val_batch_size 2048 --val_fraction 0.025 \
	--n_epochs 2 \
	--target_acc 0.7200

  r=$(( $r + 1 ))
done
