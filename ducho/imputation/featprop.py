import numpy as np 
import pandas as pd
import os 
import torch
from torch_sparse import matmul, mul, sum, fill_diag
from ducho.internal.father_classes.ImputatorFather import ImputatorFather
 
# FEATPROP (MULTIHOP): consider multi-hops relationships between items, iteratively aggregates features from neighborhood nodes.
# Performs the top-n sparsification of the item-item matrix because the projected item-item graph may be too dense (with noisy item-item interactions).
# Sets the convergence criterion through a maximum number of propagation layers

# Child class of ImputatorFather
class FeatpropImputator(ImputatorFather):

   def __init__(self, dataset_path, missing_file, input_paths, output_paths, train_path, 
                top_k, layers, modality, missing_percentage, items_file, seed):
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.input_path = input_paths
        self.output_path = output_paths
        self.train_path = train_path
        self.top_k = top_k                  # top_k: sparsification rate of the item-item matrix
        self.layers = layers                # layers: number of propagation layers
        self.modality = modality
        self.missing_percentage = missing_percentage
        self.items_file = items_file
        self.seed = seed  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   def compute_normalized_laplacian(self, adj, norm, fill_value=0.):
       adj = fill_diag(adj, fill_value=fill_value)
       deg = sum(adj, dim=-1)
       deg_inv_sqrt = deg.pow_(-norm)
       deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
       adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
       adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
       return adj_t

   def impute(self):
      result_path = os.path.join(self.output_path, "featprop")
      os.makedirs(result_path, exist_ok=True)

      if self.missing_percentage is not None:
         errors = {}
         for percentage in self.missing_percentage:
           try:
            #missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_featprop")  
            missing_items = self.compute_artificial_missing_items_new(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_featprop",  train_path=self.train_path)  
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            non_masked_embeddings_folder = f"{self.dataset_path}/non_masked_embeddings_folder_{percentage}"
            self.create_non_masked_folder(self.input_path, missing_items_npy, non_masked_embeddings_folder)
            embeddings_folder_indexed, missing_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, non_masked_embeddings_folder, missing_items, self.modality) 
            #num_items = len(missing_items) + len(os.listdir(embeddings_folder_indexed))
      
            train = pd.read_csv(train_indexed_path, sep='\t', header=None)
            num_items = train[1].nunique() #modifica
            embedding_dim = self.compute_embedding_dim(self.input_path)  
            features = torch.zeros((num_items, embedding_dim), device = self.device)
            #missing_indexed_set = set(pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
            missing_indexed_set = set(int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
            adj = self.get_item_item(train, num_items) 
            # Normalize adjacency matrix 
            adj = self.compute_normalized_laplacian(adj, 0.5) 
            for f in os.listdir(embeddings_folder_indexed):
                features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))
            missing_indexed_list = [int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist()]
            #non_missing_items = list(set(list(range(num_items))).difference(missing_indexed))
            non_missing_items = list(set(range(num_items)).difference(missing_indexed_list))

            # Compute global mean as a fallback for items that remain zero after propagation
            # The difference with respect to neighmean is that here we do the check after the propagation, checking if the final embedding is still zero
            global_mean = features[features.sum(dim=1) != 0].mean(dim=0, keepdim=False)
            propagated_features = features.clone()
            for idx in range(self.layers):
                print(f'Propagation layer: {idx + 1}')
                propagated_features = matmul(adj.to(self.device), propagated_features.to(self.device))
                propagated_features[non_missing_items] = features[non_missing_items].to(self.device)

            for miss in missing_indexed_set:
                miss_original = self.inverse_mapping(items_map, miss)
                # If the propagated embedding is still zero, use the global mean as a fallback
                if propagated_features[miss].sum() == 0:
                    np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), global_mean.detach().cpu().numpy())
                else:
                    np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), propagated_features[miss].detach().cpu().numpy())
            
            # Compute reconstruction error
            mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
            errors[percentage] = (mse, cosine_sim)

           finally: 
               self.cleanup_temp_resources([non_masked_embeddings_folder, f"{embeddings_folder_indexed}", f"{missing_indexed}", f"{train_indexed_path}"])

         return errors
      
      else:   
         missing_items = self.compute_missing_items(self.dataset_path, self.missing_file)  
         embeddings_folder_indexed, missing_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, self.input_path, missing_items, self.modality)
         #num_items = len(missing_items) + len(os.listdir(self.input_path))
 
         train = pd.read_csv(train_indexed_path, sep='\t', header=None)
         num_items = train[1].nunique() #modifica
         embedding_dim = self.compute_embedding_dim(self.input_path)
         features = torch.zeros((num_items, embedding_dim), device = self.device)
         #missing_indexed_set = set(pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
         missing_indexed_set = set(int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
         adj = self.get_item_item(train, num_items)
         # Normalize adjacency matrix 
         adj = self.compute_normalized_laplacian(adj, 0.5) 
         #torch.save(adj, "adj_featprop.pt")
         for f in os.listdir(embeddings_folder_indexed):
            features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))
         #non_missing_items = list(set(list(range(num_items))).difference(missing_indexed))
         missing_indexed_list = [int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist()]
         non_missing_items = list(set(range(num_items)).difference(missing_indexed_list))

         # Compute global mean as a fallback for items that remain zero after propagation
         global_mean = features[features.sum(dim=1) != 0].mean(dim=0, keepdim=False)
         propagated_features = features.clone()
         for idx in range(self.layers):
            print(f'Propagation layer: {idx + 1}')
            propagated_features = matmul(adj.to(self.device), propagated_features.to(self.device))
            propagated_features[non_missing_items] = features[non_missing_items].to(self.device)
         for miss in missing_indexed_set:
            miss_original = self.inverse_mapping(items_map, miss)
            # If the propagated embedding is still zero, use the global mean as a fallback
            if propagated_features[miss].sum() == 0:
               np.save(os.path.join(result_path, f'{miss_original}.npy'), global_mean.detach().cpu().numpy())
            else:
              np.save(os.path.join(result_path, f'{miss_original}.npy'), propagated_features[miss].detach().cpu().numpy())