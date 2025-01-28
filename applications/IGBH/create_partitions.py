import sys, os, time
import dgl
import torch as th
from tpp_pytorch_extension.gnn.common import gnn_utils
import torch.distributed as dist
from dgl.distgnn.dist_sampling import dist_sampling as dsam
import dgl.distgnn.dist_sampling as ds
from dgl.distgnn.partitions import partition_graph
from igb.dataloader import IGB260MDGLDataset
from igb.dataloader import IGBHeteroDGLDataset
from dgl.distgnn.mpi import init_mpi
from dgl.data.utils import save_graphs
import numpy as npy
import argparse 


def partition(args):
    graph = IGBHeteroDGLDataset(args)[0]
    tot_train = (graph.ndata['train_mask']['paper'] == True).sum()
    ind = th.nonzero(graph.ndata['train_mask']['paper'] == True, as_tuple=True)[0]
    print('##################################')
    print('Original graph: ', graph)
    print('Gloabl Train', graph.ndata['train_mask']['paper'])        
    print('Global trianing nodes: ', ind)
    print('Global training nodes: ', tot_train)
    tot_edges, tot_nodes = 0, 0
    for et in graph.etypes: tot_edges += graph.number_of_edges(etype=et)
    for nt in graph.ntypes: tot_nodes += graph.number_of_nodes(ntype=nt)
    print('Global graph edges: ', tot_edges)
    print('Global graph nodes: ', tot_nodes)
    print('##################################')
    args.part_path = os.path.join(args.part_path, args.dataset_size)
    if not os.path.exists(args.part_path):
        try:
            os.mkdir(args.part_path)
        except:
            print("Error in creating " + args.part_path)
            sys.exit(1)
    
    args.part_path = os.path.join(args.part_path, str(args.num_parts) +"p")
    if not os.path.exists(args.part_path):
        try:
            os.mkdir(args.part_path)
        except:
            print("Error in creating " + args.part_path)
            sys.exit(1)

    print('Partition location: ', args.part_path)
    node_feats = graph.ndata['feat']
    #print('graph: ', graph)
    p = partition_graph(graph,  args.num_parts, args.part_path)
    # parition_random(graph, node_feats)
    #partition_csr(graph, node_feats)
    p.partition_hetero(graph, node_feats)


if __name__ == '__main__':
    th.set_printoptions(threshold=6)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/scratch/mvasimud/omics/igb_datasets/',
        help='path containing the datasets')
    parser.add_argument('--part_path', type=str, default='/scratch/mvasimud/IGBH/partitions/',
        help='path for partitioned graph')
    
    parser.add_argument('--dataset_size', type=str, default='small',
        choices=['tiny', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--all_in_edges', type=bool, default=True,
              help="Set to false to use default relation. Set this option to True to use all the relation types in the dataset since DGL samplers require directed in edges.")
    parser.add_argument('--partition', type=bool, default=False, help="create partition or not")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='ccl', type=str,
                            help='distributed backend')
    parser.add_argument('--use_ddp', action='store_true',
                            help='use or not torch distributed data parallel')    
    parser.add_argument('--num_parts', type=int, default=4, help="how many partitions")
    parser.add_argument('--batch_size', type=int, default=1024, help="how many partitions")
    parser.add_argument('--fanout',  default=[5,10], help="how many partitions")
    args = parser.parse_args()


    partition(args)
