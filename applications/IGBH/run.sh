 python create_partitions.py --partition True --dataset_size 'medium' --num_parts 4
./run_dist.sh -n 4 -ppn 4 python -u -W ignore dist_train_rgat.py --dataset_size 'medium' --ielsqsize 0 --use_ddp --dist-backend 'ccl' --tpp_impl
