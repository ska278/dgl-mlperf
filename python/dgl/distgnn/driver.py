import sys, os, time
import dgl
import torch as th
from tpp_pytorch_extension.gnn.common import gnn_utils
import torch.distributed as dist
from dgl.distgnn.dist_sampling import dist_sampling as dsam
import dgl.distgnn.dist_sampling as ds
from igb.dataloader import IGB260MDGLDataset
from igb.dataloader import IGBHeteroDGLDataset
from dgl.distgnn.mpi import init_mpi
from dgl.data.utils import save_graphs
import numpy as npy
import argparse 

#def main():
def main(graph, node_feats, num_parts, path):
    partition(graph, node_feats, num_parts, path)


class graph_partition:
    def __init__(self, graph, etypes, ntypes, num_parts, path):
        self.num_parts = num_parts
        self.path = path
        self.etypes, self.ntypes = etypes, ntypes
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
            
            lnode_feats = {}
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
                    ## collect all types of neighbors of nodes of type nt
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
                lnode_feats[nt] = th.gather(node_feats[nt], 0, srcnodes_.to(th.int64))
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
        

        
class part_graph:
    def __init__(self, rank, num_parts):
        self.rank, self.num_parts = rank, num_parts
        
    def load_partition(self, path_, algo='csr'):
        path_ = os.path.join(path_, str(self.num_parts) +"p")
        pf = "{:04d}".format(self.rank) + ".pt"
        fol = f"part{self.rank:04d}"
        path = os.path.join(path_, fol)
        self.srcnodes = th.load(path + "/srcnodes" + pf)
        self.in_mask = th.load(path + "/in_mask" + pf)
        self.node_feats = th.load(path + "/node_feats" + pf)
        self.node_map = th.load(path + "/node_map" + pf)
        self.train_nodes = th.load(path + "/train" + pf)
        self.test_nodes = th.load(path + "/test" + pf)
        self.all_ntypes = th.load(path + "/ntypes" + pf)
        self.all_etypes = th.load(path + "/etypes" + pf)
        self.gtrain_mask = th.load(path + "/gtrain_mask" + pf)
        
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
        for nty in self.all_ntypes:
            nt = th.tensor([self.srcnodes[nty].shape[0]], dtype=th.int32)
            dist.all_reduce(nt, op=dist.ReduceOp.SUM)
            self.o2l = th.full([nt], -1, dtype=th.int32)
            self.o2lext = th.full([nt], -1, dtype=th.int32)

            self.o2l[self.srcnodes[nty]] = th.tensor([npy.arange(self.srcnodes[nty].shape[0])], dtype=th.int32)
            
            #mp = th.full(nt, -1, dtype=th.int16)
            mp = th.empty(nt, dtype=th.int16)
            st = 0
            for np in range(self.num_parts):
                ed = self.node_map[nty][np]
                mp[st : ed] = np
                st = ed

            self.node_map[nty] = mp
            #print(nty, 'node_map: ', self.node_map)

        
        
def chunkit(seeds, num_parts):
    chunk = seeds.shape[0] // num_parts
    nrem = seeds.shape[0] % num_parts
    rnge = th.tensor([ chunk + 1 if i < nrem else chunk for i in range(num_parts)], dtype=th.int32)
    final_rnge = th.cat((th.tensor([0]), th.cumsum(rnge, 0)))
    return final_rnge, rnge


def accumulate(ten):
    final_rnge = th.cat((th.tensor([0]), th.cumsum(ten, 0)))
    return final_rnge


def pitch():
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
    
    #dataset = IGB260MDGLDataset(args)
    #graph = dataset[0]
            
    if args.partition:        
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
        p = graph_partition(graph, graph.etypes, graph.ntypes, args.num_parts, args.part_path)
        # parition_random(graph, node_feats)
        #partition_csr(graph, node_feats)
        p.partition_hetero(graph, node_feats)

    else:
        ## training
        args.part_path = os.path.join(args.part_path, args.dataset_size)
        args.rank, args.world_size = init_mpi(args.dist_backend, args.dist_url)
        #print(block)
        #print(block.edges(etype='written_by'))
        rank, num_parts = args.rank, args.world_size
        p = part_graph(rank, num_parts)
        p.load_partition(args.part_path)

        seeds = th.nonzero(p.gtrain_mask == 1, as_tuple=True)[0]
        print('Total trianing nodes in the orig graph: ', seeds.size())
        acc, chunk = chunkit(seeds, num_parts)
        train_nodes = seeds[acc[rank]: acc[rank + 1]]
        print(f'[{args.rank}] Train nodes: ', train_nodes.size())
        #seeds = th.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]        
        #schunk, count = chunkit(seeds, num_parts)

        # print('p.g: ', p.g)
        fanout = args.fanout  ##[2,4]
        d = dsam(p, rank, num_parts, True)
        bs = args.batch_size ## adjust_batches(train_nodes.shape[0], args.batch_size)
        tic = time.time()
        for s in range(0, train_nodes.shape[0], bs):
        #for s in range(0, train_nodes.shape[0], bs):
            seeds = {'paper' : train_nodes[s: s+bs]}
            #print('seeds: ', seeds)            
            blocks = d.sample_neighbors(p.g, seeds, fanout)

        toc = time.time()        
        if args.rank == 0:
            print(f'Total dist sampling time: {toc-tic:.4f}')
            d.stats()
            d.timers()
            print('debug: ', ds.debug)

    #def run():
    #    args.rank, args.world_size = init_mpi(args.dist_backend, args.dist_url)
    #    rank, num_ranks = args.rank, args.world_size
    #    p = part_graph(rank, num_parts)
    #    p.load_partition_random(args.part_path)
    #    
    #    #fanout = [5,10]
    #    d = dsam(part, rank, num_parts)
    #    st, ed = schunk[rank], schunk[rank + 1]
    #    d.sample_neighbors(part.g, seeds[st:ed], fanout)

if __name__ == '__main__':

    pitch()
