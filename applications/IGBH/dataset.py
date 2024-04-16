# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import os.path as osp
import os, psutil

import dgl
from dgl.data import DGLDataset

class IGBHeteroDGLDataset(DGLDataset):
  def __init__(self,
               path,
               dataset_size='tiny',
               in_memory=False,
               use_label_2K=False,
               use_bf16=False):
    self.dir = path
    self.dataset_size = dataset_size
    self.in_memory = in_memory
    self.use_label_2K = use_label_2K
    self.use_bf16 = use_bf16

    self.ntypes = ['paper', 'author', 'institute', 'fos']
    self.etypes = None
    self.edge_dict = {}
    self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
    self.author_nodes_num = {'tiny':357041, 'small':1926066, 'medium':15544654, 'large':116959896, 'full':277220883}
    self.process()

  def process(self):
    if self.in_memory:
      paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__cites__paper', 'edge_index.npy')))
      author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__written_by__author', 'edge_index.npy')))
      affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'author__affiliated_to__institute', 'edge_index.npy')))
      paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__topic__fos', 'edge_index.npy')))
      if self.dataset_size in ['large', 'full']:
        paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'paper__published__journal', 'edge_index.npy')))
        paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'paper__venue__conference', 'edge_index.npy')))
    else:
      paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
      author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
      affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
      paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))
      if self.dataset_size in ['large', 'full']:
        paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'paper__published__journal', 'edge_index.npy'), mmap_mode='r'))
        paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
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
    if self.dataset_size in ['large', 'full']:
      self.edge_dict[('paper', 'published', 'journal')] = (paper_published_journal[:, 0], paper_published_journal[:, 1])
      self.edge_dict[('paper', 'venue', 'conference')] = (paper_venue_conference[:, 0], paper_venue_conference[:, 1])
      self.edge_dict[('journal', 'rev_published', 'paper')] = (paper_published_journal[:, 1], paper_published_journal[:, 0])
      self.edge_dict[('conference', 'rev_venue', 'paper')] = (paper_venue_conference[:, 1], paper_venue_conference[:, 0])
    self.etypes = list(self.edge_dict.keys())

    self.graph = dgl.heterograph(self.edge_dict)
    self.graph.predict = 'paper'
    print(self.graph)
    dgl.save_graphs(osp.join(self.dir, self.dataset_size, 'struct.graph'), self.graph)

    label_file = 'node_label_19.npy' if not self.use_label_2K else 'node_label_2K.npy'
    paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat.npy')
    paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)
    num_paper_nodes = self.paper_nodes_num[self.dataset_size]
    if self.in_memory:
      if self.dataset_size in ['large', 'full']:
        raise Exception(f"Cannot load related files into memory directly")
      paper_node_features = torch.from_numpy(np.load(paper_feat_path))
      if self.use_bf16:
          paper_node_features = paper_node_features.to(torch.bfloat16)
      paper_node_labels = torch.from_numpy(np.load(paper_lbl_path)).to(torch.long) 
    else:
      if self.dataset_size in ['large', 'full']:
        if self.use_bf16:
          paper_node_features = torch.from_numpy(np.memmap(paper_feat_path, dtype='float16', mode='r', shape=(num_paper_nodes,1024)))
        else:
          paper_node_features = torch.from_numpy(np.memmap(paper_feat_path, dtype='float32', mode='r', shape=(num_paper_nodes,1024)))
        paper_node_labels = torch.from_numpy(np.memmap(paper_lbl_path, dtype='float32', mode='r', shape=(num_paper_nodes))).to(torch.long)
      else:
        paper_node_features = torch.from_numpy(np.load(paper_feat_path, mmap_mode='r'))
        paper_node_labels = torch.from_numpy(np.load(paper_lbl_path, mmap_mode='r')).to(torch.long)
        if self.use_bf16:
          paper_node_features = paper_node_features.to(torch.bfloat16)

    num_author_nodes = self.author_nodes_num[self.dataset_size]
    author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat.npy')
    if self.in_memory:
      if self.dataset_size in ['large', 'full']:
        raise Exception(f"Cannot load related files into memory directly")
      author_node_features = torch.from_numpy(np.load(author_feat_path))
      if self.use_bf16:
        author_node_features = author_node_features.to(torch.bfloat16)
    else:
      if self.dataset_size in ['large', 'full']:
        if self.use_bf16:
          author_node_features = torch.from_numpy(np.memmap(author_feat_path, dtype='float16', mode='r', shape=(num_author_nodes,1024)))
        else:
          author_node_features = torch.from_numpy(np.memmap(author_feat_path, dtype='float32', mode='r', shape=(num_author_nodes,1024)))
      else:
        author_node_features = torch.from_numpy(np.load(author_feat_path, mmap_mode='r'))
        if self.use_bf16:
          author_node_features = author_node_features.to(torch.bfloat16)

    if self.in_memory:
      institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'institute', 'node_feat.npy')))
    else:
      institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'institute', 'node_feat.npy'), mmap_mode='r'))
    if self.use_bf16:
        institute_node_features = institute_node_features.to(torch.bfloat16)

    if self.in_memory:
      fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'fos', 'node_feat.npy')))
    else:
      fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'fos', 'node_feat.npy'), mmap_mode='r'))
    if self.use_bf16:
        fos_node_features = fos_node_features.to(torch.bfloat16)

    if self.dataset_size in ['large', 'full']:
      if self.in_memory:
        conference_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'conference', 'node_feat.npy')))
      else:
        conference_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'conference', 'node_feat.npy'), mmap_mode='r'))
      if self.in_memory:
        journal_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'journal', 'node_feat.npy')))
      else:
        journal_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
        'journal', 'node_feat.npy'), mmap_mode='r'))
      if self.use_bf16:
          conference_node_features = conference_node_features.to(torch.bfloat16)
          journal_node_features = journal_node_features.to(torch.bfloat16)

    graph_paper_nodes = self.graph.num_nodes('paper')
    if graph_paper_nodes < num_paper_nodes:
        self.graph.nodes['paper'].data['feat'] = paper_node_features[0:graph_paper_nodes,:]
        self.graph.num_paper_nodes = graph_paper_nodes
        self.graph.nodes['paper'].data['label'] = paper_node_labels[0:graph_paper_nodes]
    else:
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
    self.graph.nodes['author'].data['feat'] = author_node_features
    self.graph.num_author_nodes = author_node_features.shape[0]
    self.graph.nodes['institute'].data['feat'] = institute_node_features
    self.graph.num_institute_nodes = institute_node_features.shape[0]
    self.graph.nodes['fos'].data['feat'] = fos_node_features
    self.graph.num_fos_nodes = fos_node_features.shape[0]
    if self.dataset_size in ['large', 'full']:
        self.graph.num_conference_nodes = conference_node_features.shape[0]
        self.graph.nodes['conference'] = conference_node_features
        self.graph.num_journal_nodes = journal_node_features.shape[0]
        self.graph.nodes['journal'] = journal_node_features
    
    self.graph = dgl.remove_self_loop(self.graph, etype='cites')
    self.graph = dgl.add_self_loop(self.graph, etype='cites')

    if graph_paper_nodes < num_paper_nodes:
        n_nodes = graph_paper_nodes
    else:
        n_nodes = paper_node_features.shape[0]
    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    self.graph.nodes['paper'].data['train_mask'] = train_mask
    self.graph.nodes['paper'].data['val_mask'] = val_mask
    self.graph.nodes['paper'].data['test_mask'] = test_mask
    
  def __getitem__(self, i):
      return self.graph

  def __len__(self):
      return 1
