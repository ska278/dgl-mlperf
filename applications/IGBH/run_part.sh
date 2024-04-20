#!/usr/bin/bash

path=$1
dataset=$2
dataset_size=$3
in_path=$path"/"$dataset
out_path=$path"/"$dataset"/"$dataset_size
token=$5
fp_only=$6

for np in 16 #64 32 
do
    echo "Partitioning graph into "$np" parts"
    echo "Input graph from "$in_path
    echo "Output partitions into "$out_path
    if [ "x$fp_only" == "x" ]; then
      python -u -W ignore partition_graph.py --dataset $dataset --dataset_size $dataset_size --path $in_path --num_parts $np --part_method $4 --output $out_path"/"$np$token --graph_struct_only --do_partitioning --in_memory
    else
      python -u -W ignore partition_graph.py --dataset $dataset --dataset_size $dataset_size --path $in_path --num_parts $np --token $token --part_method $4 --output $out_path"/"$np$token --in_memory --feat_part_only --do_partitioning
    fi
done

