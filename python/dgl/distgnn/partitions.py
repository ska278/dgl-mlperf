import os, sys, psutil, json, argparse, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data.utils import load_tensors, load_graphs
from dgl.data import register_data_args, load_data

import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from dgl.distgnn.communicate import mpi_allreduce
from dgl.distgnn.utils import chunkit, accumulate
from dgl.data.utils import save_graphs
from igb.dataloader import IGB260MDGLDataset
from igb.dataloader import IGBHeteroDGLDataset

debug = False
'''
class partition_book:
    def __init__(self, p):
        g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes, nmi, nmp = p
        self.g_orig = g_orig
        self.node_feats = node_feats
        self.node_map = node_map
        self.num_parts = num_parts
        self.n_classes = n_classes
        self.dle = dle
        self.etypes = etypes
        self.onid_map = nmi
        self.pid_map = nmp
'''

def standardize_metis_parts(graph, node_feats, rank, resize=False):
    N = graph.number_of_nodes()
    E = graph.number_of_edges()

    if resize:
        nlocal = (graph.ndata['inner_node'] == 1).sum()
        try:
            feat = node_feats['_N/features']
        except:
            feat = node_feats['_N/feat']

        try:
            label = node_feats['_N/label'].clone().resize_(N)
        except:
            label = node_feats['_N/labels'].clone().resize_(N)
        train = node_feats['_N/train_mask'].clone().resize_(N)
        val = node_feats['_N/val_mask'].clone().resize_(N)
        test = node_feats['_N/test_mask'].clone().resize_(N)

        ten = th.zeros(N-feat.shape[0], feat.shape[1], dtype=feat.dtype)
        feat_ = th.cat((feat, ten), 0)

        train[nlocal: ] = 0
        test[nlocal: ] = 0
        val[nlocal: ] = 0
        node_feats['feat'] = feat_
        node_feats['label'] = label
        node_feats['train_mask'] = train
        node_feats['test_mask'] = test
        node_feats['val_mask'] = val

        graph.ndata['orig'] = graph.ndata['orig_id']
        del graph.ndata['orig_id']
        ntrain = (node_feats['train_mask'] == 1).sum()
        tot_train = mpi_allreduce(ntrain)

        ntest = (node_feats['test_mask'] == 1).sum()
        tot_test = mpi_allreduce(ntest)

        nval = (node_feats['val_mask'] == 1).sum()
        tot_val = mpi_allreduce(nval)

        tot_nodes = mpi_allreduce(N)
        tot_edges = mpi_allreduce(E)

        if rank == 0:
            print('tot_train nodes: ', tot_train)
            print('tot_test nodes: ', tot_test)
            print('tot_val nodes: ', tot_val)

    else:
        nlocal = (graph.ndata['inner_node'] == 1).sum()
        try:
            feat = node_feats['_N/features']
        except:
            feat = node_feats['_N/feat']

        try:
            label = node_feats['_N/label']
        except:
            label = node_feats['_N/labels']
        train = node_feats['_N/train_mask']
        val = node_feats['_N/val_mask']
        test = node_feats['_N/test_mask']

        node_feats['feat'] = feat
        node_feats['label'] = label
        node_feats['train_mask'] = train
        node_feats['test_mask'] = test
        node_feats['val_mask'] = val
        node_feats['orig'] = graph.ndata['orig_id']
        node_feats['inner_node'] = graph.ndata['inner_node']
        node_feats['orig'] = graph.ndata['orig_id']

        ntrain = (node_feats['train_mask'] == 1).sum()
        tot_train = mpi_allreduce(ntrain)

        ntest = (node_feats['test_mask'] == 1).sum()
        tot_test = mpi_allreduce(ntest)

        nval = (node_feats['val_mask'] == 1).sum()
        tot_val = mpi_allreduce(nval)

        tot_nodes = mpi_allreduce(N)
        tot_edges = mpi_allreduce(E)

        if rank == 0:
            print('tot_train nodes: ', tot_train)
            print('tot_test nodes: ', tot_test)
            print('tot_val nodes: ', tot_val)

    try:
        del node_feats['_N/feat']
    except:
        del node_feats['_N/features']
    try:
        del node_feats['_N/label']
    except:
        del node_feats['_N/labels']
    del node_feats['_N/train_mask']
    del node_feats['_N/test_mask'], node_feats['_N/val_mask']


def load_GNNdataset(args):
    dlist = ['ogbn-products', 'ogbn-papers100M']
    if args.dataset in dlist:
        assert os.path.isdir(args.path) == True
        filename = os.path.join(args.path, "struct.graph")
        tfilename = os.path.join(args.path, "tensor.pt")

        if os.path.isfile(filename) and os.path.isfile(tfilename):
            data, _ = dgl.load_graphs(filename)
            g_orig = data[0]
            n_classes = int(th.load(tfilename))
        else:
            def load_ogb(name):
                from ogb.nodeproppred import DglNodePropPredDataset
            
                data = DglNodePropPredDataset(name=name, root='./dataset')
                splitted_idx = data.get_idx_split()
                graph, labels = data[0]
                labels = labels[:, 0]
            
                graph.ndata['features'] = graph.ndata['feat']
                graph.ndata['labels'] = labels
                in_feats = graph.ndata['features'].shape[1]
                num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
            
                # Find the node IDs in the training, validation, and test set.
                train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
                train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                train_mask[train_nid] = True
                val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                val_mask[val_nid] = True
                test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
                test_mask[test_nid] = True
                graph.ndata['train_mask'] = train_mask
                graph.ndata['val_mask'] = val_mask
                graph.ndata['test_mask'] = test_mask
                return graph, num_labels

            try:
                g_orig, n_classes = load_ogb(args.dataset)
                if not debug:
                    try:
                        del g_orig.ndata['feat']
                        del g_orig.ndata['features']
                    except:
                        pass
                g_orig = dgl.add_reverse_edges(g_orig)
                if args.rank == 0 and not debug:
                    dgl.save_graphs(filename, [g_orig])
                    th.save(th.tensor(n_classes), tfilename)
            except Exception as e:
                print(e)
                n_classes = -1
                g_orig = None

    elif args.dataset == 'IGBH':
        filename = os.path.join(
                     args.path, 
                     args.dataset,
                     args.dataset_size,
                     str(args.world_size)+args.token,
                     "struct.graph"
                   )
        assert(os.path.isfile(filename)) == True
        n_classes = args.n_classes
        g_orig = dgl.load_graphs(filename)[0][0]
    else:
        print(">>>>>>>>>> Error: dataset {} not found! exiting...".format(dataset))
        sys.exit(1)

    return g_orig, n_classes


def partition_book_random(args, part_config, category='', resize_data=False):

    num_parts = args.world_size

    dls = time.time()
    g_orig, n_classes = load_GNNdataset(args)
    if category != '':
        g_orig = dgl.remove_self_loop(g_orig, etype=category)
        g_orig = dgl.add_self_loop(g_orig, etype=category)

    ntypes = g_orig.ntypes
    etypes = g_orig.canonical_etypes

    part_config_g = part_config
    di = str(num_parts) + args.token
    part_config_g = os.path.join(part_config_g, di)
    fjson = args.dataset + '.json'
    part_config = os.path.join(part_config_g, fjson)

    try:
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
    except:
        print(">>>>> Error: Partition data for {} not found!! at {}".
              format(args.dataset, part_config))
        sys.exit(1)

    prefix = part_config_g + "/"
    part_files = part_metadata['part-{}'.format(args.rank)]
    node_feats = load_tensors(prefix + part_files['node_feats'])
    graph = load_graphs(prefix + part_files['part_graph'])[0][0]
    
    path = prefix + part_files['part_graph']
    dname = os.path.dirname(path)
    node_map_index = th.load(os.path.join(dname, "node_map_index.pt"))  ## Push these into json
    node_map_part = th.load(os.path.join(dname, "node_map_part.pt"))  ## Push these into json
    
    nf_keys = node_feats.keys()

    num_nodes_per_ntype = [g_orig.num_nodes(ntype) for ntype in g_orig.ntypes]
    offset_per_ntype = np.insert(np.cumsum(num_nodes_per_ntype), 0, 0)

    nsn = int(graph.ndata['inner_node'].sum())

    for keys, nt in enumerate(g_orig.ntypes):
        ft = nt + '/feat'
        if keys == 0:
            node_feats['feat'] = node_feats[ft]
        else:
            node_feats['feat'] = th.cat((node_feats['feat'], node_feats[ft]), 0)
        del node_feats[ft]
    if args.use_bf16 and (not node_feats['feat'].dtype == th.bfloat16): 
        node_feats['feat'] = node_feats['feat'].to(th.bfloat16)
    elif not args.use_bf16 and node_feats['feat'].dtype == th.bfloat16:
        node_feats['feat'] = node_feats['feat'].to(th.float32)
    dle = time.time() - dls

    acc, acc_labels = 0, 0
    masks = []
    label = []
    for keys in nf_keys:
        if 'mask' in keys:
            masks.append(keys)
        if 'label' in keys:
            label.append(keys)

    node_feats['train_mask'] = th.zeros(nsn, dtype=th.uint8)
    #node_feats['test_mask']  = th.zeros(nsn, dtype=th.uint8)
    node_feats['val_mask']   = th.zeros(nsn, dtype=th.uint8)
    node_feats['labels']     = th.zeros(nsn, dtype=th.int64)

    for keys, nt in enumerate(g_orig.ntypes):
        pnid = (graph.ndata['_TYPE'][:nsn] == keys).nonzero(as_tuple=True)[0]
        for m in masks:
            if nt in m:
                if 'train' in m:
                    node_feats['train_mask'][pnid] = node_feats[m]
                    acc += node_feats[m].sum()
                elif 'val' in m:
                    m = nt+'/val_mask'
                    node_feats['val_mask'][pnid] = node_feats[m]
                #elif 'test' in m:
                #    m = nt+'/test_mask'
                #    node_feats['test_mask'][pnid] = node_feats[m]
        for l in label:
            if nt in l:
                node_feats['labels'][pnid] = node_feats[l]
                acc_labels += (node_feats[l] > 0).sum()

    node_feats['inner_node'] = graph.ndata['inner_node']
    node_feats['orig'] = graph.ndata['orig_id']
    node_feats['_TYPE'] = graph.ndata['_TYPE']
    node_feats['train_samples'] = g_orig.ndata['train_mask']['paper'].sum()
    node_feats['eval_samples'] = g_orig.ndata['val_mask']['paper'].sum()
                
    assert acc ==  node_feats['paper/train_mask'].sum()
    assert acc_labels == (node_feats['paper/label'] > 0).sum()
    del graph
    del node_feats['paper/train_mask']
    del node_feats['paper/val_mask']
    del node_feats['paper/test_mask']
    del node_feats['paper/label']
    del g_orig.ndata['test_mask']['paper']
    del g_orig.ndata['train_mask']['paper']
    del g_orig.ndata['val_mask']['paper']

    etypes = g_orig.etypes

    #node_map = part_metadata['node_map']
    node_map = None
    d = g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes, node_map_index, node_map_part
    pb = partition_book(d)
    return pb


def partition_book_metis(args, part_config, resize_ndata=False):

    num_parts = args.world_size

    dls = time.time()
    g_orig, n_classes = load_GNNdataset(args)
    ntypes = g_orig.ntypes
    etypes = g_orig.canonical_etypes

    part_config_g = part_config

    di = args.dataset + "-" + str(num_parts) + args.token + "-balance-train"
    part_config_g = os.path.join(part_config_g, di)
    fjson = args.dataset + '.json'
    part_config = os.path.join(part_config_g, fjson)

    #if args.rank == 0:
    #    print("Dataset/partition location: ", part_config)

    try:
        with open(part_config) as conf_f:
            part_metadata = json.load(conf_f)
    except:
        print(">>>>> Error: Partition data for {} not found!! at {}".
              format(args.dataset, part_config))
        sys.exit(1)

    prefix = part_config_g + "/"
    part_files = part_metadata['part-{}'.format(args.rank)]
    assert 'node_feats' in part_files, "the partition does not contain node features."
    assert 'edge_feats' in part_files, "the partition does not contain edge feature."
    assert 'part_graph' in part_files, "the partition does not contain graph structure."
    node_feats = load_tensors(prefix + part_files['node_feats'])
    graph = load_graphs(prefix + part_files['part_graph'])[0][0]

    dle = time.time() - dls

    #num_parts = part_metadata['num_parts']
    #node_map_  = part_metadata['node_map']

    standardize_metis_parts(graph, node_feats, args.rank, resize_ndata)
    del graph
    if args.use_bf16: node_feats['feat'] = node_feats['feat'].to(th.bfloat16)

    #node_map = []   ## this block should go in standardize_metis_parts
    #for nm in node_map_['_N']:
    #    node_map.append(nm[1])

    node_map = None
    if args.rank == 0:
        print("n_classes: ", n_classes, flush=True)

    etypes = None
    d = g_orig, node_feats, node_map, num_parts, n_classes, dle, etypes, None, None
    pb = partition_book(d)
    return pb


def create_partition_book(args, part_config, resize=False):
    if args.part_method == 'metis':
        pb = partition_book_metis(args, part_config, resize)
    elif args.part_method == 'random':
        pb = partition_book_random(args, part_config, resize)

    return pb



class args_:
    def __init__(self, dataset):
        self.dataset = dataset
        print("dataset set to: ", self.dataset)



def main(graph, node_feats, num_parts, path):
    partition(graph, node_feats, num_parts, path)

    
###########################################################################################
## horizontal division of orig CSR graph martix to create partitions
class partition_graph:
    def __init__(self, graph, num_parts, path):
        self.num_parts = num_parts
        self.path = path
        self.etypes, self.ntypes = graph.etypes, graph.ntypes
        self.canon_edges = {}
        for nt in self.ntypes:
            self.canon_edges[nt] = ''
        for t in graph.canonical_etypes:
            try:
                self.canon_edges[t[2]].append(t)
            except:
                self.canon_edges[t[2]] = [t]

        self.etype_dt = {}
        self.canon_etypes = []
        #for i, e in enumerate(self.etypes):
        for i, t in enumerate(graph.canonical_etypes):
            self.etype_dt[t] = i
            self.canon_etypes.append(t)
        
    def to_csr(self, graph, dir='in'):
        r'''
        converts the coo to csr matrix
        '''
        
        u, v = self.graph.edges()
        uu, index = th.sort(u)
        indices = v[index]
        srcnodes, count = uu.unique(sorted=True, return_counts=True)
        indptr = th.cat((th.tensor([0]), th.cumsum(count)[:-1]))
        return (srcnodes, indptr, indices)

    
    #def to_coo(self, graph, nt, dir='in'):
    def partition_hetero(self, graph, node_feats, method='flat', edir='in'):
        print('ndata: ', graph.ndata)
        train_mask = graph.ndata['train_mask']['paper']
        test_mask = graph.ndata['test_mask']['paper']
        glabels = graph.ndata['label']['paper']
        
        g, train_all, test_all = [], [], []
        for rank in range(self.num_parts):
            print(f'Partitioning for rank {rank}')
            num_nodes = {}
            num_edges = {}

            for k,v in node_feats.items():
                lnode_feats = th.tensor([], dtype=v.dtype)
                break
            
            node_map = {}
            srcnodes = {}
            pg = dgl.DGLGraph()
            aa = th.tensor([], dtype=th.int32)
            bb = th.tensor([], dtype=th.int32)
            cc = th.tensor([], dtype=th.int32)
            in_mask = {}
            
            train = test = 0, 0
            for nt in self.ntypes:
                in_mask[nt] = th.full([graph.number_of_nodes(ntype=nt)], 0, dtype=th.bool)
                u = th.tensor([], dtype=th.int32)
                v = th.tensor([], dtype=th.int32)
                e = th.tensor([], dtype=th.int32)
                # e = []
                # e, count = [], []
                nodes = graph.nodes(ntype=nt)
                rnge, count = chunkit(nodes, self.num_parts)
                node_map[nt] = rnge[1:]
                #print('canon_edges: ', self.canon_edges)

                if self.canon_edges[nt] != '': 
                    ## collect all types of neighbors of node type nt
                    for et in self.canon_edges[nt]:
                        #print('et: ', et)
                        if et == '': continue
                        vv, uu = graph.in_edges(nodes, etype=et)
                        u = th.cat((u,uu))
                        v = th.cat((v,vv))
                        #print('et: ', et)
                        #print('etype_dt: ', self.etype_dt)
                        e = th.cat((e, th.tensor([self.etype_dt[et] for i in range(uu.shape[0])], \
                                                 dtype=e.dtype)))
                        #e = e + [et for i in range(uu.shape[0])]
                        #e.append(et)
                        #count.append(uu.shape[0])
                    
                    a, indices = th.sort(u)
                    b, c = v[indices], e[indices]
                    k, count = th.unique(a, return_counts=True)

                    ##### grouping k nodes according to equal division of 'nodes'
                    s, e = rnge[rank], rnge[rank + 1]
                    ind1 = k >= s #th.nonzero(k >= s, as_tuple=True)[0]
                    ind2 = k < e  #th.nonzero(k < e, as_tuple=True)[0]
                    ind  = th.nonzero(ind1 & ind2, as_tuple=True)[0]
                    #print('k: ', k, s, e)
                    #print('ind1: ', ind1)
                    #print('ind2: ', ind2)
                    #print('ind: ', ind)
                    s, e = ind[0], ind[-1] + 1
                    in_mask[nt][k[s]:k[e-1] + 1] = 1  ## check
                    if nt == 'fos':
                        assert k.max() < 190449

                    #rnge, node_map[nt] = chunkit(k, self.num_parts)
                    #print('nt: ', nt, ' #nodes: ', nodes.shape[0], ' k: ', k[s:e], k[s:e].size(), 'nm: ', rnge)
                    count_acc = accumulate(count)
                    s, e = count_acc[s], count_acc[e]  ## edges  ## check
                    
                    if nt == 'paper':
                        s1, e1 = rnge[rank], rnge[rank + 1]
                        #print('getting taining nodes: ', s1, e1)
                        train = train_mask[s1 : e1]
                        train = th.nonzero(train, as_tuple=True)[0] + rnge[rank]
                        labels = glabels[train]
                        test = test_mask[s1: e1]
                        test = th.nonzero(test, as_tuple=True)[0] + rnge[rank]
                    
                    ## Append edges for all ntypes for a given rank
                    aa = th.cat((aa, a[s:e]))
                    bb = th.cat((bb, b[s:e]))
                    cc = th.cat((cc, c[s:e]))
                    #print('a: ', a[s:e], a[s:e].size(), s, e)
                
                l, m = rnge[rank], rnge[rank+1]
                srcnodes[nt] = nodes[l:m]
                srcnodes_ = srcnodes[nt].unsqueeze(1).repeat(1,node_feats[nt].shape[1])
                #lnode_feats[nt] = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
                lfeats = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
                lnode_feats = th.cat((lnode_feats, lfeats), 0)
                #print('et: ', et)
                #print('node_map: ', node_map)
                #print('srcnodes: ', srcnodes_, srcnodes_.size())
                #print('nt: ', nt, 'node_feats ', node_feats[nt].size())
                #print('##################################################')
            ## for loop of ntype ends

            #print("'''''''''''''''''''''''''''''''''''''")
            #print(f'[{rank:04d}] train nodes: ', train, train.size())
            c, indices = th.sort(cc)
            a, b = aa[indices], bb[indices]
            k, count = th.unique(c, return_counts=True)
            #num_nodes[]
            count_acc = accumulate(count)
            print(f'[{rank:04d}] total edges: ', count_acc[-1])
            gdt = {}
            for i, kk in enumerate(k):
                s, e = count_acc[i], count_acc[i+1]
                #pg.add_edges(self.etype_dt[kk]:(a[s:e], b[s:e]))
                gdt[self.canon_etypes[kk] ] = (b[s:e], a[s:e])
                print('--', self.canon_etypes[kk], a[s:e].max(), b[s:e].max())
                
            #srt, _ = th.sort(a)  
            #srcnodes = th.unique(srt)
            #srcnodes_ = srcnodes.unsqueeze(1).repeat(1,node_feats[nt].shape[1])
            
            #lnode_feats = gnn_utils.gather_features(node_feats, srcnodes.to(th.long))
            #lnode_feats[nt] = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
            
            print(gdt)
            pg = dgl.heterograph(gdt)
            print('pg: ', pg)
            ##print('fos max node: ', pg.nodes(ntype='fos').max())
            ##assert pg.nodes(ntype='fos').max() < 190449
            #g.append(pg)
            #train_all.append(train)
            #test_all.append(test)
            pf = "{:04d}".format(rank) + ".pt"
            try:
                fol = f"part{rank:04d}"
                os.mkdir(fol)
            except:
                pass
              
            path = os.path.join(self.path, fol)
            save_graphs(path + "/graph.dgl", pg)
            th.save(lnode_feats, path + "/node_feats" + pf)
            th.save(srcnodes, path + "/srcnodes" + pf)
            th.save(in_mask, path + "/in_mask" + pf)
            th.save(node_map, path + "/node_map" + pf)
            
            th.save(train, path + "/train" + pf)
            th.save(test, path + "/test" + pf)
            th.save(self.ntypes, path + "/ntypes" + pf)
            th.save(self.etypes, path + "/etypes" + pf)
            th.save(graph.ndata['train_mask']['paper'], path + "/gtrain_mask" + pf)
            th.save(graph.ndata['test_mask']['paper'], path + "/gtest_mask" + pf)
            th.save(graph.ndata['label']['paper'], path + "/glabels" + pf)
            
        print(f"Graph partition completed and saved at {path}.")
        
    # Function to extract rows
    def extract_csr_rows(self, csr_matrix, extract_rows):
        # Access CSR components
        rows, indptr, indices = csr_matrix
        
        # Extract new row pointers, values, and column indices
        new_rows = []
        new_indices = []
        new_indptr = [0]
        nnz = 0  # Track non-zero count
        
        for row in extract_rows:
            start = indptr[row]
            end = indptr[row + 1]
            new_rows.extend(rows[start:end].tolist())
            new_indices.extend(indices[start:end].tolist())
            nnz += end - start
            new_indptr.append(nnz)
        
        # Convert to tensors
        new_rows = th.tensor(new_rows, dtype=rows.dtype)
        new_indices = th.tensor(new_indices, dtype=indices.dtype)
        new_indptr = th.tensor(new_indptr, dtype=indptr.dtype)
        
        # Create the new CSR matrix
        #new_shape = (len(rows), csr_matrix.size(1))
        #return csr_tensor((new_values, new_col_indices, new_row_pointers), size=new_shape)
        return new_rows, new_indptr, new_indices


    #######################################################     
    def partition_hetero_random(self, graph, node_feats, method='random', edir='in'):
        print('ndata: ', graph.ndata)
        train_mask = graph.ndata['train_mask']['paper']
        test_mask = graph.ndata['test_mask']['paper']
        glabels = graph.ndata['label']['paper']
        
        g, train_all, test_all = [], [], []
        for rank in range(self.num_parts):
            print(f'Partitioning for rank {rank}')
            num_nodes = {}
            num_edges = {}

            for k,v in node_feats.items():
                lnode_feats = th.tensor([], dtype=v.dtype)
                break
            
            node_map = {}
            srcnodes = {}
            pg = dgl.DGLGraph()
            aa = th.tensor([], dtype=th.int32)
            bb = th.tensor([], dtype=th.int32)
            cc = th.tensor([], dtype=th.int32)
            in_mask = {}
            
            train = test = 0, 0
            for nt in self.ntypes:
                in_mask[nt] = th.full([graph.number_of_nodes(ntype=nt)], 0, dtype=th.bool)
                u = th.tensor([], dtype=th.int32)
                v = th.tensor([], dtype=th.int32)
                e = th.tensor([], dtype=th.int32)
                # e = []
                # e, count = [], []
                nodes = graph.nodes(ntype=nt)
                rnge, count = chunkit(nodes, self.num_parts)
                node_map[nt] = rnge[1:]
                #print('canon_edges: ', self.canon_edges)

                if self.canon_edges[nt] != '': 
                    ## collect all types of neighbors of node type nt
                    for et in self.canon_edges[nt]:
                        #print('et: ', et)
                        if et == '': continue
                        vv, uu = graph.in_edges(nodes, etype=et)
                        u = th.cat((u,uu))
                        v = th.cat((v,vv))
                        #print('et: ', et)
                        #print('etype_dt: ', self.etype_dt)
                        e = th.cat((e, th.tensor([self.etype_dt[et] for i in range(uu.shape[0])], \
                                                 dtype=e.dtype)))
                        #e = e + [et for i in range(uu.shape[0])]
                        #e.append(et)
                        #count.append(uu.shape[0])
                    
                    a, indices = th.sort(u)
                    b, c = v[indices], e[indices]
                    k, count = th.unique(a, return_counts=True)

                    ##### grouping k nodes according to equal division of 'nodes'
                    s, e = rnge[rank], rnge[rank + 1]
                    ind1 = k >= s #th.nonzero(k >= s, as_tuple=True)[0]
                    ind2 = k < e  #th.nonzero(k < e, as_tuple=True)[0]
                    ind  = th.nonzero(ind1 & ind2, as_tuple=True)[0]
                    #print('k: ', k, s, e)
                    #print('ind1: ', ind1)
                    #print('ind2: ', ind2)
                    #print('ind: ', ind)
                    s, e = ind[0], ind[-1] + 1
                    in_mask[nt][k[s]:k[e-1] + 1] = 1  ## check
                    if nt == 'fos':
                        assert k.max() < 190449

                    #rnge, node_map[nt] = chunkit(k, self.num_parts)
                    #print('nt: ', nt, ' #nodes: ', nodes.shape[0], ' k: ', k[s:e], k[s:e].size(), 'nm: ', rnge)
                    count_acc = accumulate(count)
                    s, e = count_acc[s], count_acc[e]  ## edges  ## check
                    
                    if nt == 'paper':
                        s1, e1 = rnge[rank], rnge[rank + 1]
                        #print('getting taining nodes: ', s1, e1)
                        train = train_mask[s1 : e1]
                        train = th.nonzero(train, as_tuple=True)[0] + rnge[rank]
                        labels = glabels[train]
                        test = test_mask[s1: e1]
                        test = th.nonzero(test, as_tuple=True)[0] + rnge[rank]
                    
                    ## Append edges for all ntypes for a given rank
                    aa = th.cat((aa, a[s:e]))
                    bb = th.cat((bb, b[s:e]))
                    cc = th.cat((cc, c[s:e]))
                    #print('a: ', a[s:e], a[s:e].size(), s, e)
                
                l, m = rnge[rank], rnge[rank+1]
                srcnodes[nt] = nodes[l:m]
                srcnodes_ = srcnodes[nt].unsqueeze(1).repeat(1,node_feats[nt].shape[1])
                #lnode_feats[nt] = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
                lfeats = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
                lnode_feats = th.cat((lnode_feats, lfeats), 0)
                #print('et: ', et)
                #print('node_map: ', node_map)
                #print('srcnodes: ', srcnodes_, srcnodes_.size())
                #print('nt: ', nt, 'node_feats ', node_feats[nt].size())
                #print('##################################################')
            ## for loop of ntype ends

            #print("'''''''''''''''''''''''''''''''''''''")
            #print(f'[{rank:04d}] train nodes: ', train, train.size())
            c, indices = th.sort(cc)
            a, b = aa[indices], bb[indices]
            k, count = th.unique(c, return_counts=True)
            #num_nodes[]
            count_acc = accumulate(count)
            print(f'[{rank:04d}] total edges: ', count_acc[-1])
            gdt = {}
            for i, kk in enumerate(k):
                s, e = count_acc[i], count_acc[i+1]
                #pg.add_edges(self.etype_dt[kk]:(a[s:e], b[s:e]))
                gdt[self.canon_etypes[kk] ] = (b[s:e], a[s:e])
                print('--', self.canon_etypes[kk], a[s:e].max(), b[s:e].max())
                
            #srt, _ = th.sort(a)  
            #srcnodes = th.unique(srt)
            #srcnodes_ = srcnodes.unsqueeze(1).repeat(1,node_feats[nt].shape[1])
            
            #lnode_feats = gnn_utils.gather_features(node_feats, srcnodes.to(th.long))
            #lnode_feats[nt] = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
            
            print(gdt)
            pg = dgl.heterograph(gdt)
            print('pg: ', pg)
            ##print('fos max node: ', pg.nodes(ntype='fos').max())
            ##assert pg.nodes(ntype='fos').max() < 190449
            #g.append(pg)
            #train_all.append(train)
            #test_all.append(test)
            pf = "{:04d}".format(rank) + ".pt"
            try:
                fol = f"part{rank:04d}"
                os.mkdir(fol)
            except:
                pass
              
            path = os.path.join(self.path, fol)
            save_graphs(path + "/graph.dgl", pg)
            th.save(lnode_feats, path + "/node_feats" + pf)
            th.save(srcnodes, path + "/srcnodes" + pf)
            th.save(in_mask, path + "/in_mask" + pf)
            th.save(node_map, path + "/node_map" + pf)
            
            th.save(train, path + "/train" + pf)
            th.save(test, path + "/test" + pf)
            th.save(self.ntypes, path + "/ntypes" + pf)
            th.save(self.etypes, path + "/etypes" + pf)
            th.save(graph.ndata['train_mask']['paper'], path + "/gtrain_mask" + pf)
            th.save(graph.ndata['test_mask']['paper'], path + "/gtest_mask" + pf)
            th.save(graph.ndata['label']['paper'], path + "/glabels" + pf)
            
        print(f"Graph partition completed and saved at {path}.")

    
    def partition_random(self, graph, node_feats):
        r'''
        Goal: partitions coo edges randomly
        '''
        '''
        for r in range(self.num_parts):
            try:
                fol = "part{r:04d}"
                os.mkdir(fol)
            except:
                pass
            
            path = os.path.join(self.path, fol)            
            a, b,  = th.tensor([]), th.tensor([]), th.tensor([])
            for et in self.etypes:
                u, v = self.graph.edges(etype = et)
                acc, count = chunkit(u, self.num_parts)
                a = th.cat((a, u[acc[r]: acc[r+1]]))
                b = th.cat((b, v[acc[r]: acc[r+1]]))
                c = th.cat((c, th.full([et], u.shape[0], dtype=u.dtype)))
    
            pf = "{:04d}".format(r) + ".pt"            
            th.save((a, b, c), path + "/coo" + pf)
            
            for nt in self.ntypes:
                nodes = graph.nodes(ntype=nt)
                acc, count = chunkit(nodes, self.num_parts)
                fnode = nodes[acc[r]: acc[r+1]]
                lnode_feats = gnn_utils.gather_features(node_feats[nt], fnodes.to(th.long))        
                th.save(lnode_feats, path + "/node_feats" + pf)        
        '''

        srcnodes, indptr, indices = to_csr(graph)
        #rows = th.tensor(np.arange(indptr.shape[0] - 1), dtype=th.int32)
        chunk = indptr.shape[0] // self.num_parts
        nrem = indptr.shape[0] % self.num_parts
        nodes_ar = [ chunk + 1 if i < nrem else chunk for i in range(self.num_parts)]
        nodes_ar = th.tensor(nodes_ar, dtype=th.int32)
        node_map = th.cat((th.tensor([0]), th.cumsum(nodes_ar)))
        feat_size = node_feats.shape[1]

        csr_matrix = srcnodes, indptr, indices            
        randoms = torch.randint(0, self.num_parts, (indptr.shape[0]))
        for rank in range(self.num_parts):
            ind = th.nonzero(randoms == rank, as_tuple=True)[0]
            extract_rows = randoms[ind]
            lsrcnodes, lindptr, lindices = self.extract_csr_rows(csr_matrix, extract_rows)
            
            n = ind.shape[0]
            lnode_feats = gnn_utils.gather_features(node_feats, lsrcnodes.to(th.long))
            #node_feats = th.gather(nodes_feats, 0, lsrcnodes.to(th.int64))
            pf = "{:04d}".format(rank) + ".pt"
            try:
                fol = "part{rank:04d}"
                os.mkdir(fol)
            except:
                pass
    
            path = os.path.join(self.path, fol)
    
            u = th.empty(n, dtype=th.int32)
            v = th.empty(n, dtype=th.int32)
            st = lindptr[0]
            for i in range(lindptr.shape[0]):
                ed = lindptr[i+1]
                u[st:ed], v[st:ed] = lsrcnodes[i], lindices[st: ed]            
                st = ed
                
            g = dgl.DGLGraph((u, v))
            dgl.save_graph(path + "/graph.dgl", g)
            th.save(lsrcnodes, path + "/srcnodes" + pf)
            th.save(lindptr, path + "/indptr" + pf)
            th.save(lindices, path + "/indices" + pf)
            th.save(lnode_feats, path + "/node_feats" + pf)
            th.save(node_map, path + "/node_map" + pf)

            th.save(train_mask, path + "/train_mask" + pf)
            th.save(test_mask, path + "/test_mask" + pf)
            
            print(f"[{rank:04}] Wrote out the partitions at {path}")
            
        print(f"Graph partition completed and saved at {path}.")

            
            
    def partition_csr(self, graph, node_feats):
        r'''
        Partitions the CSR matrix horizontally
        '''

        srcnodes, indptr, indices = self.to_csr(graph)
        chunk = indptr.shape[0] // self.num_parts
        nrem = indptr.shape[0] % self.num_parts
        nodes_ar = th.tensor([ chunk + 1 if i < nrem else chunk \
                               for i in range(self.num_parts)], dtype=th.int32)
        #nodes_ar = th.tensor(nodes_ar, dtype=th.int32)
        node_map = th.cat((th.tensor([0]), th.cumsum(nodes_ar)))
        feat_size = node_feats.shape[1]
        ## acc_chunks, chunks = chunkit(indptr.shape[0], self.num_parts)
        #ind = th.nonzero(graph.ndata['train_mask']['paper'], as_tuple=True)[0]
        
        for rank in range(self.num_parts):
            a, b = node_map[rank], node_map[rank + 1]
            
            train_mask = graph.ndata['train_mask']['paper'][a:b-1]
            ind = th.nonzero(train_mask, as_tuple=True)[0]
            train_nodes = train_mask[ind] + node_map[rank]
            
            test_mask = graph.ndata['test_mask'][a:b-1]
            ind = th.nonzero(test_mask, as_tuple=True)[0]
            test_nodes = test_mask[ind] + node_map[rank]
            
            #val_mask = graph.ndata['val_mask'][a:b-1]
            lsrcnodes = srcnodes[a:b - 1]            
            lindptr = indptr[a:b]
            lindices = indices[lindptr[0]: lindptr[-1]]
            
            n = b - a
            lnode_feats = gnn_utils.gather_features(node_feats, lsrcnodes.to(th.long))
            #node_feats = th.gather(nodes_feats, 0, lsrcnodes.to(th.int64))
            pf = "{:04d}".format(rank) + ".pt"
            try:
                fol = "part{rank:04d}"
                os.mkdir(fol)
            except:
                pass
    
            path = os.path.join(self.path, fol)
    
            u = th.empty(n, dtype=th.int32)
            v = th.empty(n, dtype=th.int32)
            st = lindptr[0]
            for i in range(lindptr.shape[0]):
                ed = lindptr[i+1]
                u[st:ed], v[st:ed] = lsrcnodes[i], lindices[st: ed]            
                st = ed
                
            g = dgl.DGLGraph((u, v))
            dgl.save_graph(path + "/graph.dgl", g)
            th.save(lsrcnodes, path + "/srcnodes" + pf)
            th.save(lindptr, path + "/indptr" + pf)
            th.save(lindices, path + "/indices" + pf)
            th.save(train_mask, path + "/train_nodes" + pf)
            th.save(test_mask, path + "/test_nodes" + pf)
            th.save(lnode_feats, path + "/node_feats" + pf)
            th.save(node_map, path + "/node_map" + pf)
            print(f"[{rank:04}] Wrote out the partitions at {path}")
            
        print(f"Graph partition completed and saved at {path}.")
        #print("Node feats are not saved/processed.")
        

### New parition book
class partition_book:
    def __init__(self, rank, num_parts, part_config, gformat='csr'):
        self.rank, self.num_parts = rank, num_parts
        tic = time.time()
        self.load_partition(part_config, gformat)
        self.dle = time.time() - tic
        
    def load_partition(self, path_, gformat):
        path_ = os.path.join(path_, str(self.num_parts) +"p")
        pf = "{:04d}".format(self.rank) + ".pt"
        fol = f"part{self.rank:04d}"
        path = os.path.join(path_, fol)
        #self.num_nodes_orig = th.load(path + "/num_nodes_orig" + pf)
        self.srcnodes = th.load(path + "/srcnodes" + pf)
        self.in_mask = th.load(path + "/in_mask" + pf)

        self.node_feats = {}
        self.node_feats['feats'] = th.load(path + "/node_feats" + pf)
        self.node_map = th.load(path + "/node_map" + pf)
        ###### Note: not used for now
        self.train_nodes = th.load(path + "/train" + pf)
        self.test_nodes = th.load(path + "/test" + pf)
        ######
        '''
        self.node_feats['all_ntypes'] = th.load(path + "/ntypes" + pf)
        self.node_feats['all_etypes'] = th.load(path + "/etypes" + pf)
        self.node_feats['srcnodes'] = th.load(path + "/srcnodes" + pf)
        ## Not all srcnodes expect incoming edges
        self.node_feats['in_mask'] = th.load(path + "/in_mask" + pf)   
        '''
        self.all_ntypes = th.load(path + "/ntypes" + pf)
        self.all_etypes = th.load(path + "/etypes" + pf)
        self.gtrain_mask = th.load(path + "/gtrain_mask" + pf)
        self.gtest_mask = th.load(path + "/gtest_mask" + pf)
        self.glabels = th.load(path + "/glabels" + pf)        
        
        self.g = dgl.load_graphs(path + "/graph.dgl")[0][0]
        self.setup()
        #return


    ## old, defunct, now load_partition_csr can load both type of parititons
    def load_partition_random(self, path_):
        r = self.rank
        pf = "{:04d}".format(self.rank) + ".pt"
        fol = f"part{self.rank:04d}"
        path = os.path.join(path_, fol)            
        self.node_feats = th.load(path + "/node_feats" + pf)
        self.graph_coo = th.load(path + "/coo" + pf)
        self.setup()
        
        
    def setup(self):
        self.num_nodes_orig = {}
        for nty in self.all_ntypes:
            nt = th.tensor([self.srcnodes[nty].shape[0]], dtype=th.int32)
            dist.all_reduce(nt, op=dist.ReduceOp.SUM)
            self.num_nodes_orig[nty] = int(nt)
            
            #self.o2l = th.full([nt], -1, dtype=th.int32)
            #self.o2lext = th.full([nt], -1, dtype=th.int32)
            #
            #self.o2l[self.srcnodes[nty]] = th.tensor([np.arange(self.srcnodes[nty].shape[0])], dtype=th.int32)
            
            #mp = th.full(nt, -1, dtype=th.int16)
            mp, st = th.empty(nt, dtype=th.int16), 0            
            for npt in range(self.num_parts):
                ed = self.node_map[nty][npt]
                mp[st : ed] = npt
                st = ed

            self.node_map[nty] = mp
            #print(nty, 'node_map: ', self.node_map)

        ## concat the srcnodes offseted with num_nodes_orig[nt]
        ## put the concat nodes in node_feats['orig]
        offset = 0
        for k, v in self.srcnodes.items():
            aten  = th.tensor([], dtype=v.dtype)
            break
        
        for i, nt in enumerate(self.all_ntypes):
            #self.ntypes_id[nt] = keys
            #self.acc_onodes[nt] = offset
            #try:
            aten = th.cat((aten, self.srcnodes[nt] + offset))
            #except:
            #    pass
            offset += self.num_nodes_orig[nt]
            #if self.rank == 0:
            #    print('node_map: ', self.node_map)
            #    print('assert: ', (self.node_map[nt][self.srcnodes[nt]] == self.rank).sum(), self.srcnodes[nt].shape[0])
            #assert int((self.node_map[nt][self.srcnodes[nt]] == self.rank).sum()) == self.srcnodes[nt].shape[0]
            
        self.node_feats['orig'] = aten
        #print('paritions.py: aten: ', aten)
        print('Total original graph nodes: ', offset, self.num_nodes_orig)
        #print('Accumulate orig grpah nodes: ', self.)
        
