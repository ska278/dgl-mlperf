#!/bin/bash

r=1
while [ $r -le $1 ]
do
  run_dist.sh -n 32 -ppn 2 -f /home/savancha/GNN/distgnn-master/applications/igb/hostfile \
	python -u -W ignore mn_rgat_train.py \
	--path /data/savancha \
	--dataset IGBH --dataset_size full \
	--dist-backend mpi --mode iels --opt_mlp \
       	--use_bf16 --n_classes 2983 \
	--learning_rate $2 \
	--batch_size 1024 \
	--val_batch_size 1024 \
        --val_fraction 0.025 \
	--hidden_channels 512 \
	--log_every 36 \
	--n_epochs 2 \
	--target_acc 0.7200

  r=$(( $r + 1 ))
done
