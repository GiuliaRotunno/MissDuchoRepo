# metodi per l'imputazione a grafo
import numpy as np
import pandas as pd
import os
import torch
import scipy
import scipy.sparse as sp 
from torch_sparse import SparseTensor
from torch_sparse import SparseTensor, mul, sum, fill_diag

def get_item_item(self, train, num_items):  # compute item_item matrix
       user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())), shape=(train[0].nunique(), num_items), dtype=np.float32)
       item_item = user_item.transpose().dot(user_item).toarray()
       knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=self.device), self.top_k, dim=-1)
       items_cols = torch.flatten(knn_ind).to(self.device)
       ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=self.device)
       items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
       final_adj = SparseTensor(row=items_rows, col=items_cols, value=torch.tensor([1.0] * items_rows.shape[0], device=self.device), sparse_sizes=(item_item.shape[0], item_item.shape[0]))
       return final_adj
   
def mapping(dataset_path, train_path, embeddings_folder, missing_items, modality):
       train_path = dataset_path + f'/{train_path}'
       train = pd.read_csv(train_path, sep='\t', header=None) 
       users = train[0].unique()
       items = train[1].unique()
       users_map = {u: idx  for idx, u in enumerate(users)}
       items_map = {i: idx  for idx, i in enumerate(items)}
       train[0] = train[0].map(users_map)
       train[1] = train[1].map(items_map)
       train_indexed_path = dataset_path + f'/train_indexed.tsv'
       train.to_csv(train_indexed_path, sep='\t', index=False, header=None)
      
       embeddings_folder_indexed = embeddings_folder + f'_indexed'
       os.makedirs(embeddings_folder_indexed, exist_ok=True)
       for key, value in items_map.items():
           if os.path.exists(os.path.join(embeddings_folder, f'{key}.npy')):
              np.save(f'{embeddings_folder_indexed}/{value}.npy', np.load(f'{embeddings_folder}/{key}.npy'))

       missing_indexed = dataset_path + f'/missing_{modality}_indexed.tsv'
       with open(missing_indexed, 'w') as f:
            for i in missing_items:
                for key, value in items_map.items():        
                    if i == key:
                      f.write(str(value))
                      f.write('\n') 

       return embeddings_folder_indexed, missing_indexed, items_map, train_indexed_path
   
def inverse_mapping(items_map, item_mapped):
       inverse_map = {v: k for k, v in items_map.items()}
       item_original = inverse_map.get(item_mapped)
       return item_original

def compute_normalized_laplacian(adj, norm, fill_value=0.): #featprop
       adj = fill_diag(adj, fill_value=fill_value)
       deg = sum(adj, dim=-1)
       deg_inv_sqrt = deg.pow_(-norm)
       deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
       adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
       adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
       return adj_t

def compute_normalized_pagerank(self, adj, num_items): #perspagerank
       adj = sp.coo_matrix((adj.storage._value.cpu().numpy(), (adj.storage._row.cpu().numpy(), adj.storage._col.cpu().numpy())), shape=(num_items, num_items))
       num_nodes = num_items
       A_tilde = adj + np.eye(num_nodes)
       D_tilde = 1 / np.sqrt(A_tilde.sum(axis=1))
       D_ = np.zeros((num_items, num_items))
       np.fill_diagonal(D_, D_tilde[0, 0])
       H = D_ @ A_tilde @ D_
       adj = self.alpha * np.linalg.inv(np.eye(num_items) - (1 - self.alpha) * H)
       row_idx = np.arange(num_nodes)
       adj[adj.argsort(axis=0)[:num_nodes - self.top_k], row_idx] = 0.
       norm = adj.sum(axis=0)
       norm[norm <= 0] = 1
       adj = adj / norm
       return adj 
     
def compute_normalized_heat(self, adj, num_items): #heat
       adj = sp.coo_matrix((adj.storage._value.cpu().numpy(), (adj.storage._row.cpu().numpy(), adj.storage._col.cpu().numpy())), shape=(num_items, num_items))
       num_nodes = num_items
       A_tilde = adj + np.eye(num_nodes)
       D_tilde = 1 / np.sqrt(A_tilde.sum(axis=1))
       D_ = np.zeros((num_items, num_items))
       np.fill_diagonal(D_, D_tilde[0, 0])
       H = D_ @ A_tilde @ D_
       adj = scipy.linalg.expm(-self.time * (np.eye(num_nodes) - H))
       row_idx = np.arange(num_nodes)
       adj[adj.argsort(axis=0)[:num_nodes - self.top_k], row_idx] = 0.
       norm = adj.sum(axis=0)
       norm[norm <= 0] = 1
       adj = adj / norm
       return adj



  