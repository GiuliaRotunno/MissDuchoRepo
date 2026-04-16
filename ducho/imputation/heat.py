import numpy as np
import pandas as pd
import os
import torch
import scipy
import scipy.sparse as sp
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# HEAT: modifies the MULTIHOP ('featprop') imputation using heat diffusion as a normalisation of the item-item cointeractions matrix 
# the projected item-item graph may be too dense (with noisy item-item interactions) -> perform the top-n sparsification of the item-item matrix
# set the convergence criterion through a maximum number of propagation layers

#classe figlia di ImputatorFather
class HeatImputator(ImputatorFather):  #Apply heat imputation for missing items
   """
      Args: missing_items (List[str]): List of missing item IDs.
            output_path (str): Path to save imputed embeddings.
            layers (int): Number of propagation layers, values in [1,2,3,20].
            top_k (int): Sparsification rate of the item-item matrix, values in [10, 20, ... 100].
            time (int): Diffusion time (also 'b'), values in -----------. 
   """
   def __init__(self, dataset_path, missing_file, input_paths, output_paths, train_path, top_k, layers, time, modality, missing_percentage, items_file, seed):
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.input_path = input_paths
        self.output_path = output_paths
        self.train_path = train_path
        self.top_k = top_k
        self.layers = layers
        self.time = time
        self.modality = modality
        self.missing_percentage = missing_percentage
        self.items_file = items_file
        self.seed = seed 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

   def impute(self):
      result_path = os.path.join(self.output_path, "heat")
      os.makedirs(result_path, exist_ok=True)

      if self.missing_percentage is not None:
         errors = {}
         for percentage in self.missing_percentage:
           try: 
            missing_items = self.compute_artificial_missing_items_new(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_heat",  train_path=self.train_path) 
            #missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_heat")  
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            #non_masked_embeddings_folder = f"{self.dataset_path}/non_masked_embeddings_folder_{percentage}"
            non_masked_embeddings_folder = f"{self.dataset_path}/non_masked_embeddings_folder_{percentage}_h"
            self.create_non_masked_folder(self.input_path, missing_items_npy, non_masked_embeddings_folder)
            embeddings_folder_indexed, missing_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, non_masked_embeddings_folder, missing_items, self.modality) #funzione in graph_imputation
            #num_items = len(missing_items) + len(os.listdir(embeddings_folder_indexed))
            
            train = pd.read_csv(train_indexed_path, sep='\t', header=None)
            num_items = train[1].nunique() #modifica
            embedding_dim = self.compute_embedding_dim(self.input_path)  #funzione in simple_imputation
            features = torch.zeros((num_items, embedding_dim), device = self.device)
            #missing_indexed_set = set(pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
            missing_indexed_set = set(int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
            adj = self.get_item_item(train, num_items) #funzione in graph_imputation
            adj = self.compute_normalized_heat(adj, num_items) # normalize adjacency matrix #funzione in graph_imputation
            
            for f in os.listdir(embeddings_folder_indexed):
                features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))           
            #non_missing_items = list(set(list(range(num_items))).difference(missing_indexed))
            missing_indexed_list = [int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist()] #new
            non_missing_items = list(set(range(num_items)).difference(missing_indexed_list))

            # Calcola media globale come fallback per item che rimangono a zero dopo la propagazione
            global_mean = features[features.sum(dim=1) != 0].mean(dim=0).detach().cpu().numpy()
            features = features.detach().cpu().numpy()
            propagated_features = features.copy()
            for idx in range(self.layers):
               print(f'Propagation layer: {idx + 1}')
               #propagated_features = np.matmul(adj, propagated_features)
               propagated_features = np.array(adj) @ propagated_features
               propagated_features[non_missing_items] = features[non_missing_items]
               
            for miss in missing_indexed_set:
                miss_original = self.inverse_mapping(items_map, miss)
                # Se l'embedding propagato è ancora zero, usa la global mean come fallback
                if propagated_features[miss].sum() == 0:
                    np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), global_mean)
                else:
                    np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), propagated_features[miss])
            
            # Calcola errore di ricostruzione
            mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
            errors[percentage] = (mse, cosine_sim)
             
           finally: 
               self.cleanup_temp_resources([non_masked_embeddings_folder, f"{embeddings_folder_indexed}", f"{missing_indexed}", f"{train_indexed_path}"])

         return errors

      else:
         missing_items = self.compute_missing_items(self.dataset_path, self.missing_file)  #funzione in simple_imputation
         embeddings_folder_indexed, missing_indexed, items_map, train_indexed_path = self.mapping(self.dataset_path, self.train_path, self.input_path, missing_items, self.modality) #funzione in graph_imputation
         #num_items = len(missing_items) + len(os.listdir(self.input_path))
       
         train = pd.read_csv(train_indexed_path, sep='\t', header=None)
         num_items = train[1].nunique() #modifica
         embedding_dim = self.compute_embedding_dim(self.input_path)  #funzione in simple_imputation
         features = torch.zeros((num_items, embedding_dim), device = self.device)
         #missing_indexed_set = set(pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
         missing_indexed_set = set(int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist())
         adj = self.get_item_item(train, num_items) #funzione in graph_imputation
         adj = self.compute_normalized_heat(adj, num_items) # normalize adjacency matrix #funzione in graph_imputation
         #torch.save(adj, "adj_heat.pt")
         for f in os.listdir(embeddings_folder_indexed):
             features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))
         #non_missing_items = list(set(list(range(num_items))).difference(missing_indexed))
         missing_indexed_list = [int(x) for x in pd.read_csv(missing_indexed, sep='\t', header=None)[0].tolist()]
         non_missing_items = list(set(range(num_items)).difference(missing_indexed_list))

         # Calcola media globale come fallback per item che rimangono a zero dopo la propagazione
         global_mean = features[features.sum(dim=1) != 0].mean(dim=0).detach().cpu().numpy()
         features = features.detach().cpu().numpy()
         propagated_features = features.copy()
         for idx in range(self.layers):
            print(f'Propagation layer: {idx + 1}')
            #propagated_features = np.matmul(adj, propagated_features)
            propagated_features = np.array(adj) @ propagated_features
            propagated_features[non_missing_items] = features[non_missing_items]
         for miss in missing_indexed_set:
            miss_original = self.inverse_mapping(items_map, miss)
            # Se l'embedding propagato è ancora zero, usa la global mean come fallback
            if propagated_features[miss].sum() == 0:
               np.save(os.path.join(result_path, f'{miss_original}.npy'), global_mean)
            else:
               np.save(os.path.join(result_path, f'{miss_original}.npy'), propagated_features[miss])
         
        
         """
            features = features.detach().cpu().numpy()
            propagated_features = features.copy()
            for idx in range(self.layers):
               print(f'Propagation layer: {idx + 1}')
               propagated_features = np.matmul(adj, propagated_features)
               propagated_features[non_missing_items] = features[non_missing_items]
            for miss in missing_indexed_set:
                miss_original = inverse_mapping(items_map, miss)  # mapping inverso per salvarli con la stringa di partenza #funzione in graph_imputation
                np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), propagated_features[miss])
            """
        
         """
         features = features.detach().cpu().numpy()
         propagated_features = features.copy()
         for idx in range(self.layers):
             print(f'Propagation layer: {idx + 1}')
             propagated_features = np.matmul(adj, propagated_features)
             propagated_features[non_missing_items] = features[non_missing_items]
         for miss in missing_indexed_set:
             miss_original = inverse_mapping(items_map, miss)  # mapping inverso per salvarli con la stringa di partenza #funzione in graph_imputation
             np.save(os.path.join(result_path, f'{miss_original}.npy'), propagated_features[miss])
         """
          


     
