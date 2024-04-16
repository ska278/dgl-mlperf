import argparse
import os
import sys
import time

import dgl
from dgl.data import DGLDataset

import numpy as np
import torch 
import os.path as osp
from tpp_pytorch_extension.gnn.common import gnn_utils

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        self.process_graph()

    def process_graph(self):

        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))

            if args.dataset_size in ['large', 'full']:
                paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__published__journal', 'edge_index.npy')))
                paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__venue__conference', 'edge_index.npy')))

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))
            if args.dataset_size in ['large', 'full']:
                paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__published__journal', 'edge_index.npy'), mmap_mode='r'))
                paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
                'paper__venue__conference', 'edge_index.npy'), mmap_mode='r'))

        self.edge_dict = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0])
        }
        if args.dataset_size in ['large', 'full']:
            self.edge_dict[('paper', 'published', 'journal')] = (paper_published_journal[:, 0], paper_published_journal[:, 1])
            self.edge_dict[('paper', 'venue', 'conference')] = (paper_venue_conference[:, 0], paper_venue_conference[:, 1])
            self.edge_dict[('journal', 'rev_published', 'paper')] = (paper_published_journal[:, 1], paper_published_journal[:, 0])
            self.edge_dict[('conference', 'rev_venue', 'paper')] = (paper_venue_conference[:, 1], paper_venue_conference[:, 0])
        self.etypes = list(self.edge_dict.keys())

        self.graph = dgl.heterograph(self.edge_dict)     
        self.graph.predict = 'paper'

        graph_paper_nodes = self.graph.num_nodes('paper')
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')

        if args.dataset_size == 'full':
            n_nodes = 157675969
        else:
            n_nodes = graph_paper_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
  
        shuffled_index = torch.randperm(n_nodes)      
        self.train_idx = shuffled_index[:n_train]
        self.val_idx = shuffled_index[n_train : n_train + n_val]
        self.test_idx = shuffled_index[n_train + n_val:]
        del shuffled_index

        train_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        val_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph_paper_nodes, dtype=torch.bool)
        
        train_mask[self.train_idx] = True
        val_mask[self.val_idx] = True
        test_mask[self.test_idx] = True

        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="dataset path",
    )
    argparser.add_argument(
        "--dataset_size",
        type=str,
        help="dataset size: tiny, medium, large, full"
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--in_memory",
        action="store_true",
        help="load into memory",
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--do_partitioning",
        action="store_true",
        help="split only features across partitions",
    )
    argparser.add_argument(
        "--feat_part_only",
        action="store_true",
        help="split only features across partitions",
    )
    argparser.add_argument(
        "--graph_struct_only",
        action="store_true",
        help="split only features across partitions",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    torch.manual_seed(111269)
    dgl.seed(128203)
    start = time.time()
    if args.graph_struct_only:
        g = IGBHeteroDGLDataset(args)[0]
        dgl.save_graphs(osp.join(args.path, args.dataset_size, str(args.num_parts)+"p", 'struct.graph'), g)
        print(
            "load {} takes {:.3f} seconds".format(args.path, time.time() - start)
        )
        print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
        print(
            "train: {}, valid: {}, test: {}".format(
                torch.sum(g.ndata["train_mask"][g.predict]),
                torch.sum(g.ndata["val_mask"][g.predict]),
                torch.sum(g.ndata["test_mask"][g.predict]),
            )
        )
    elif args.feat_part_only:
        g = dgl.load_graphs(osp.join(args.path, args.dataset_size, str(args.num_parts)+"p", 'struct.graph'))[0][0]
        print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))

        label_file = 'node_label_2K.npy'
        if not (args.dataset_size in ['large', 'full']):
           paper_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat.npy')
           paper_lbl_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', label_file)
           paper_node_features = torch.from_numpy(np.load(paper_feat_path))
           paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).to(torch.long)

           author_feat_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat.npy')
           author_node_features = torch.from_numpy(np.load(author_feat_path))
           institute_node_features = torch.from_numpy(np.load(osp.join(args.path, args.dataset_size, 'processed',
                                     'institute', 'node_feat.npy')))
           fos_node_features = torch.from_numpy(np.load(osp.join(args.path, args.dataset_size, 'processed',
                               'fos', 'node_feat.npy')))
           g.nodes['paper'].data['feat'] = paper_node_features
           g.num_paper_nodes = paper_node_features.shape[0]
           g.nodes['paper'].data['label'] = paper_node_labels
           g.nodes['author'].data['feat'] = author_node_features
           g.num_author_nodes = author_node_features.shape[0]
           g.nodes['institute'].data['feat'] = institute_node_features
           g.num_institute_nodes = institute_node_features.shape[0]
           g.nodes['fos'].data['feat'] = fos_node_features
           g.num_fos_nodes = fos_node_features.shape[0]
        else:
            paper_lbl_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', label_file)
            paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'paper', 'node_feat.pt')
            graph_paper_nodes = g.num_nodes('paper')
            if args.dataset_size == 'full':
                g.nodes['paper'].data['feat'] = torch.load(feat_path)[:graph_paper_nodes]
                g.nodes['paper'].data['label'] = paper_node_labels[:graph_paper_nodes]
            else:
                g.nodes['paper'].data['feat'] = torch.load(feat_path)
                g.nodes['paper'].data['label'] = paper_node_labels

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'author', 'node_feat.pt')
            g.nodes['author'].data['feat'] = torch.load(feat_path)

            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'institute', 'node_feat.pt')
            g.nodes['institute'].data['feat'] = torch.load(feat_path)
            
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'fos', 'node_feat.pt')
            g.nodes['fos'].data['feat'] = torch.load(feat_path)
            
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'conference', 'node_feat.pt')
            g.nodes['conference'].data['feat'] = torch.load(feat_path)
            
            feat_path = osp.join(args.path, args.dataset_size, 'processed', 'journal', 'node_feat.pt')
            g.nodes['journal'].data['feat'] = torch.load(feat_path)

            #g.num_author_nodes = author_node_features.shape[0]
            #g.num_institute_nodes = institute_node_features.shape[0]
            #g.num_fos_nodes = fos_node_features.shape[0]

            #g.num_conference_nodes = conference_node_features.shape[0]
            #g.num_journal_nodes = journal_node_features.shape[0]

    if args.do_partitioning:
        cores = int(os.environ["OMP_NUM_THREADS"])
        gnn_utils.affinitize_cores(cores, 0)

        dgl.distributed.partition_graph(
            g,
            args.dataset,
            args.num_parts,
            args.output,
            num_hops=1,
            part_method=args.part_method,
            balance_ntypes=None,
            balance_edges=args.balance_edges,
            return_mapping=False,
            num_trainers_per_machine=args.num_trainers_per_machine,
            feat_part_only=args.feat_part_only
        )
        #torch.save(orig_nids, osp.join(args.path, args.dataset_size, str(args.num_parts)+"p", 'orig_nids.dgl'))
        #torch.save(orig_eids, osp.join(args.path, args.dataset_size, str(args.num_parts)+"p", 'orig_eids.dgl'))

