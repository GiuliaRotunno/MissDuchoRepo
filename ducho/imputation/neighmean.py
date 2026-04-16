import numpy as np
import pandas as pd
import os
import torch
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# NEIGHMEAN: extends the 'mean' imputation through the mean of multimodal features from the one-hop neighborhood
# the projected item-item graph may be too dense (with noisy item-item interactions) -> perform the top-n sparsification of the item-item matrix

#classe figlia di ImputatorFather
class NeighmeanImputator(ImputatorFather):  #Apply neighmean imputation for missing items
   """   
      Args: missing_items (List[str]): List of missing item IDs.
            output_path (str): Path to save imputed embeddings.
            top_k (int): Sparsification rate of the item-item matrix, values in [10, 20, ... 100].
   """ 
   def __init__(self, dataset_path, missing_file, input_paths, output_paths, train_path, top_k, modality, missing_percentage, items_file, seed): 
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.input_path = input_paths
        self.output_path = output_paths
        self.train_path = train_path
        self.top_k = top_k
        self.modality = modality
        self.missing_percentage = missing_percentage
        self.items_file = items_file
        self.seed = seed       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
   def impute(self): 
      result_path = os.path.join(self.output_path, "neighmean")
      os.makedirs(result_path, exist_ok=True)

      if self.missing_percentage is not None: 
         errors = {}
         for percentage in self.missing_percentage:
           try:
            #missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_neighmean")  #funzione in simple_imputation
            missing_items = self.compute_artificial_missing_items_new(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_neighmean",  train_path=self.train_path)  
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            non_masked_embeddings_folder = f"{self.dataset_path}/non_masked_embeddings_folder_{percentage}_n"
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
            for f in os.listdir(embeddings_folder_indexed):
                features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))
            
            #MODIFICA (per calcolo cosine similarity)
            # Filtra i vicini escludendo quelli con embedding ancora a zero (item anch'essi mancanti).
            # Questo evita di calcolare medie su vettori nulli che causerebbero cosine similarity = 0.0
            # La norma zero si verifica quando i vicini di un item mancante sono anch'essi mancanti, quindi non hanno embedding caricati e rimangono inizializzati a zero.
            
            # Calcola media globale come fallback per item senza vicini validi
            global_mean = features[features.sum(dim=1) != 0].mean(dim=0, keepdim=True)
            for miss in missing_indexed_set:
               first_hop = adj[miss].storage._col
               valid_neighbors = first_hop[features[first_hop].sum(dim=1) != 0] # FILTRA i vicini che hanno embedding validi (non zero)
               if len(valid_neighbors) == 0:
                  mean_ = global_mean   # Nessun vicino con embedding valido -> usa global mean
               else:
                 # Calcola la media solo sui vicini con embedding validi
                 mean_ = features[valid_neighbors].mean(axis=0, keepdims=True)
                 #print(f"Item {miss}: {len(valid_neighbors)} vicini validi su {len(first_hop)} totali")
               miss_original = self.inverse_mapping(items_map, miss)
               np.save(os.path.join(result_path_percentage, f'{miss_original}.npy'), mean_.detach().cpu().numpy())

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
         for f in os.listdir(embeddings_folder_indexed):
            features[int(f.split('.npy')[0]), :] = torch.from_numpy(np.load(os.path.join(embeddings_folder_indexed, f)))
         
         # Calcola media globale come fallback per item senza vicini validi
         global_mean = features[features.sum(dim=1) != 0].mean(dim=0, keepdim=True)
         for miss in missing_indexed_set:
            first_hop = adj[miss].storage._col
            valid_neighbors = first_hop[features[first_hop].sum(dim=1) != 0] # FILTRA i vicini che hanno embedding validi (non zero)
            if len(valid_neighbors) == 0:
               mean_ = global_mean   # Nessun vicino con embedding valido -> usa global mean
            else:
               # Calcola la media solo sui vicini con embedding validi
               mean_ = features[valid_neighbors].mean(axis=0, keepdims=True)
               #print(f"Item {miss}: {len(valid_neighbors)} vicini validi su {len(first_hop)} totali")
            miss_original = self.inverse_mapping(items_map, miss)
            np.save(os.path.join(result_path, f'{miss_original}.npy'), mean_.detach().cpu().numpy())



         """
            for miss in missing_indexed_set:
               first_hop = adj[miss].storage._col
               mean_ = features[first_hop].mean(axis=0, keepdims=True)  
               miss_original = inverse_mapping(items_map, miss)  # mapping inverso per salvarli con la stringa di partenza  #funzione in graph_imputation
               np.save(os.path.join(result_path_percentage or result_path, f'{miss_original}.npy'), mean_.detach().cpu().numpy())
         """