import pandas as pd 
import numpy as np
import os
from loguru import logger
import glob
import torch
import scipy.sparse as sp 
from torch_sparse import SparseTensor
from tqdm import tqdm
import random
import shutil

class ImputatorFather:

    def __init__(self):
        pass 

    ### Functions for simple imputation

    # Function to infer the embeddings dimension
    def compute_embedding_dim(self, input_path):
      
      embedding_files = glob.glob(os.path.join(input_path, '*.npy'))
      # To infer dimensions, use the first file found
      if embedding_files:
         sample_embedding = np.load(embedding_files[0])
         embedding_dim = sample_embedding.shape[1]
         logger.info(f"Inferred embedding dimension: {embedding_dim}")
      else:
         raise ValueError("Could not infer embedding dimension. No .npy files found for any model.")
      
      return embedding_dim
    
    # Function that returns the list of items with missing modalities
    def compute_missing_items(self, dataset_path, missing_file):
         
         missing_file_path = os.path.join(dataset_path, missing_file)
         if not os.path.exists(missing_file_path):
            logger.error(f"Missing file not found: {missing_file_path}")
         logger.info(f"Applying imputation using missing data from {missing_file_path}")

         # The missing_file is a text file with one item per line
         try:
              with open(missing_file_path, 'r') as f:
                    missing_data = f.readlines()
                    missing_items = [line.strip() for line in missing_data]
         except Exception as e:
              logger.error(f"Error loading missing data file: {str(e)}")

         return missing_items

    # Function to create an artificial list of items with missing modalities
    def compute_artificial_missing_items(self, dataset_path, missing_percentage, items_file, seed, file_name):

        if missing_percentage < 0.0 or missing_percentage > 1.1:
            logger.error(f"Percentage of missing items must be between 0.0 and 0.99")
        else:
            items_file_path = os.path.join(dataset_path, items_file)
            if not os.path.exists(items_file_path):
                    logger.error(f"Items file not found: {items_file_path}")
            logger.info(f"Computing artificial missing items")
            # Read the TSV file with items having all modalities 
            df = pd.read_csv(items_file_path, sep="\t")           
            items = df["item_name"].tolist()     
            # Seed for reproducibility                 
            np.random.seed(seed)
            # Create visual masks for items
            mask  = np.random.rand(len(items)) < missing_percentage 
            if not mask.any(): 
                # Guarantees at least one masked item per modality
                mask[np.random.randint(len(items))] = True  

            artificial_missing_file = f'{dataset_path}/{file_name}.tsv'
            with open(artificial_missing_file, "w") as f:
                for i, item in enumerate(items):
                    if mask[i]:
                        f.write(f"{item}\n")

            with open(artificial_missing_file, 'r') as f:
                artificial_missing_data = f.readlines()
                artificial_missing_items = [line.strip() for line in artificial_missing_data]
                
        return artificial_missing_items


    ### Functions for graph imputation

    # Function to create an artificial list of items with missing modalities (with train_path)
    def compute_artificial_missing_items_new(self, dataset_path, missing_percentage, items_file, seed, file_name, train_path=None):

        if missing_percentage < 0.0 or missing_percentage > 1.1:
            logger.error(f"Percentage of missing items must be between 0.0 and 0.99")
        else:
            items_file_path = os.path.join(dataset_path, items_file)
            if not os.path.exists(items_file_path):
                logger.error(f"Items file not found: {items_file_path}")
            logger.info(f"Computing artificial missing items")
            df = pd.read_csv(items_file_path, sep="\t")
            items = df["item_name"].tolist()

            if train_path is not None:
                train = pd.read_csv(f'{dataset_path}/{train_path}', sep='\t', header=None)
                
                # Load the items that actually have an embedding
                items_with_embeddings = set([f.split('.npy')[0] for f in os.listdir(self.input_path)])
                # For each item, find its neighbors in the training set and verify that at least one has an embedding
                item_neighbors = train.groupby(1)[0].apply(set).to_dict()  # item -> set of users
                user_items = train.groupby(0)[1].apply(set).to_dict()      # user -> set of items
                
                def has_neighbor_with_embedding(item):
                    if item not in item_neighbors:
                        return False
                    users = item_neighbors[item]
                    for user in users:
                        for neighbor in user_items[user]:
                            if str(neighbor) in items_with_embeddings and str(neighbor) != str(item):
                                return True
                    return False
                
                items = [item for item in items if has_neighbor_with_embedding(item)]
                logger.info(f"Items with at least one neighbor with embedding: {len(items)}")

            np.random.seed(seed)
            mask = np.random.rand(len(items)) < missing_percentage
            if not mask.any():
                mask[np.random.randint(len(items))] = True

            artificial_missing_file = f'{dataset_path}/{file_name}.tsv'
            with open(artificial_missing_file, "w") as f:
                for i, item in enumerate(items):
                    if mask[i]:
                        f.write(f"{item}\n")

            with open(artificial_missing_file, 'r') as f:
                artificial_missing_data = f.readlines()
                artificial_missing_items = [line.strip() for line in artificial_missing_data]

        return artificial_missing_items

    # Function to compute the item-item matrix
    def get_item_item(self, train, num_items):
       
       user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())), 
                                shape=(train[0].nunique(), num_items), dtype=np.float32)
       item_item = user_item.transpose().dot(user_item).toarray()
       knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=self.device), self.top_k, dim=-1)
       items_cols = torch.flatten(knn_ind).to(self.device)
       ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=self.device)
       items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
       final_adj = SparseTensor(row=items_rows, col=items_cols, value=torch.tensor([1.0] * items_rows.shape[0], device=self.device), 
                                sparse_sizes=(item_item.shape[0], item_item.shape[0]))
       return final_adj

    # Function to compute the id-number mapping 
    def mapping(self, dataset_path, train_path, embeddings_folder, missing_items, modality):
       
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

    # Function to compute the number-id inverse mapping 
    def inverse_mapping(self,items_map, item_mapped):

        inverse_map = {v: k for k, v in items_map.items()}
        item_original = inverse_map.get(item_mapped)

        return item_original

    # Function to create a folder with non-masked items 
    def create_non_masked_folder(self, input_path, missing_items_npy, folder_path):
   
        os.makedirs(folder_path, exist_ok=True)
        for file in os.listdir(input_path):
            if file.endswith('.npy') and (file not in missing_items_npy): 
                np.save(f'{folder_path}/{file}', np.load(os.path.join(input_path, file)))
    

    ### Functions for Autoencoder-based imputation

    # Function to set the seed for reproducibility
    def set_seed(self):
      
       random.seed(self.seed)
       np.random.seed(self.seed)
       torch.manual_seed(self.seed)
       torch.cuda.manual_seed_all(self.seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

    # Function to normalize features and compute the global means
    def normalize_features(self, visual_features, textual_features):
       
        # Normalize visual features
        visual_tensor = torch.from_numpy(visual_features)
        visual_min = visual_tensor.min(0, keepdim=True)[0]
        visual_max = visual_tensor.max(0, keepdim=True)[0]
        visual_normalized = (visual_tensor - visual_min) / (visual_max - visual_min + 1e-8)
        visual_normalized = visual_normalized.to(self.device)
        
        # Normalize textual features
        textual_tensor = torch.from_numpy(textual_features)
        textual_min = textual_tensor.min(0, keepdim=True)[0]
        textual_max = textual_tensor.max(0, keepdim=True)[0]
        textual_normalized = (textual_tensor - textual_min) / (textual_max - textual_min + 1e-8)
        textual_normalized = textual_normalized.to(self.device)
        
        # Move to device
        visual_min = visual_min.to(self.device)
        visual_max = visual_max.to(self.device)
        textual_min = textual_min.to(self.device)
        textual_max = textual_max.to(self.device)
        
        # Compute global means as fallback
        visual_global_mean = visual_tensor.mean(dim=0).to(self.device)
        textual_global_mean = textual_tensor.mean(dim=0).to(self.device)
        
        return (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean)

    # Load visual and textual features, and find the items with both modalities
    def load_features(self, visual_folder, textual_folder):
        
        visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape 
        num_items_visual = len(os.listdir(visual_folder))
        visual_items = os.listdir(visual_folder)
        
        textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape
        num_items_textual = len(os.listdir(textual_folder))
        textual_items = os.listdir(textual_folder)
        
        all_present = [x for x in textual_items if x in visual_items]
        visual_features = np.empty((len(all_present), visual_shape[-1]), dtype=np.float32) if num_items_visual else None
        textual_features = np.empty((len(all_present), textual_shape[-1]), dtype=np.float32) if num_items_textual else None
        
        for idx, it in enumerate(all_present):
            visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
            textual_features[idx, :] = np.load(os.path.join(textual_folder, it))
        
        return visual_features, textual_features, visual_shape, textual_shape

    # Load indexed features, and find the items with both modalities
    def load_indexed_features(self, visual_embeddings_folder_indexed, textual_embeddings_folder_indexed, missing_visual_indexed, missing_textual_indexed):
        
        visual_shape = np.load(os.path.join(visual_embeddings_folder_indexed, os.listdir(visual_embeddings_folder_indexed)[0])).shape
        num_items_visual = len(os.listdir(visual_embeddings_folder_indexed))
        visual_items = os.listdir(visual_embeddings_folder_indexed)

        textual_shape = np.load(os.path.join(textual_embeddings_folder_indexed, os.listdir(textual_embeddings_folder_indexed)[0])).shape
        num_items_textual = len(os.listdir(textual_embeddings_folder_indexed))
        textual_items = os.listdir(textual_embeddings_folder_indexed)

        all_present = [int(x.split('.npy')[0]) for x in textual_items if x in visual_items]
        visual_features = np.empty((len(all_present), visual_shape[-1]), dtype=np.float32) if num_items_visual else None
        textual_features = np.empty((len(all_present), textual_shape[-1]), dtype=np.float32) if num_items_textual else None
        inner_map_item = {}

        for idx, it in enumerate(all_present): 
            inner_map_item[it] = idx
            visual_features[idx, :] = np.load(os.path.join(visual_embeddings_folder_indexed, f'{it}.npy'))
            textual_features[idx, :] = np.load(os.path.join(textual_embeddings_folder_indexed, f'{it}.npy'))

        return visual_features, textual_features, visual_shape, textual_shape, all_present, inner_map_item

    # Prepare the adjacency matrix from the training set 
    def prepare_adjacency_matrix(self, train_indexed_path, all_present):
        
        train = pd.read_csv(train_indexed_path, sep='\t', header=None)
        train = train[train[1].isin(all_present)]
        inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}
        inner_map_item = {item: idx for idx, item in enumerate(all_present)}
        train[0] = train[0].map(inner_map_user)
        train[1] = train[1].map(inner_map_item)
        adj = self.get_item_item(train, num_items=len(all_present))

        return adj


    ### Functions for Deep learning-based imputation

    # Function to load data and identify the items with both modalities 
    def load_data(self, visual_folder, textual_folder):

        # Identify the items with both modalities 
        visual_items = set(os.listdir(visual_folder))
        textual_items = set(os.listdir(textual_folder))
        # List of filenames with both modalities
        all_present = sorted(list(visual_items & textual_items)) 
        
        # Load features for complete items
        visual_features = []
        textual_features = []
        for item in tqdm(all_present, desc="Loading complete items"):
            visual_path = os.path.join(visual_folder, item)
            textual_path = os.path.join(textual_folder, item)
            visual_feat = np.load(visual_path)
            textual_feat = np.load(textual_path)
            if visual_feat.ndim == 2 and visual_feat.shape[0] == 1:
               visual_feat = visual_feat.squeeze(0)
            if textual_feat.ndim == 2 and textual_feat.shape[0] == 1:
                textual_feat = textual_feat.squeeze(0)
            visual_features.append(visual_feat)
            textual_features.append(textual_feat)
        visual_features = np.array(visual_features, dtype=np.float32)    # Numpy array [N, visual_dim]
        textual_features = np.array(textual_features, dtype=np.float32)  # Numpy array [N, textual_dim]
        
        return visual_features, textual_features, all_present

    # Function to normalize features in [0,1]
    def normalize_data(self, visual_features, textual_features): 

        visual_tensor = torch.from_numpy(visual_features).float()
        textual_tensor = torch.from_numpy(textual_features).float()

        # Compute min/max
        self.visual_min = visual_tensor.min(dim=0, keepdim=True)[0]
        self.visual_max = visual_tensor.max(dim=0, keepdim=True)[0]
        self.textual_min = textual_tensor.min(dim=0, keepdim=True)[0]
        self.textual_max = textual_tensor.max(dim=0, keepdim=True)[0]

        # Calculate global means as a fallback for items with both modalities missing
        self.visual_global_mean = visual_tensor.mean(dim=0)
        self.textual_global_mean = textual_tensor.mean(dim=0)

        # Normalize
        visual_norm = (visual_tensor - self.visual_min) / (self.visual_max - self.visual_min + 1e-8)
        textual_norm = (textual_tensor - self.textual_min) / (self.textual_max - self.textual_min + 1e-8)

        # Move to device
        visual_norm = visual_norm.to(self.device)
        textual_norm = textual_norm.to(self.device)
        self.visual_min = self.visual_min.to(self.device)
        self.visual_max = self.visual_max.to(self.device)
        self.textual_min = self.textual_min.to(self.device)
        self.textual_max = self.textual_max.to(self.device)
        self.visual_global_mean = self.visual_global_mean.to(self.device)
        self.textual_global_mean = self.textual_global_mean.to(self.device)

        return visual_norm, textual_norm

    # Function to denormalize from [0,1] to the original range
    def denormalize(self, data, modality='visual'):
        if modality == 'visual':
            return data * (self.visual_max - self.visual_min) + self.visual_min
        else: # if modality == 'textual'
            return data * (self.textual_max - self.textual_min) + self.textual_min
    

    ### Functions to compute reconstruction errors

    # Function to compute MSE and Cosine similarity
    def compute_reconstruction_errors(self, real_embeddings, imputed_embeddings, eps=1e-8):
      mse = []
      cosine_sim = []

      for real_emb, imp_emb in zip(real_embeddings, imputed_embeddings):
         real_emb = np.asarray(real_emb).reshape(-1)
         imp_emb = np.asarray(imp_emb).reshape(-1)

        # DEBUG
         if np.isnan(real_emb).any() or np.isnan(imp_emb).any():
            print(f"----- NaN found in the vectors!")
            print(f"  real NaN: {np.isnan(real_emb).sum()}, imp NaN: {np.isnan(imp_emb).sum()}")
            continue  # skip this sample

         if np.isinf(real_emb).any() or np.isinf(imp_emb).any():
            print(f"----- Inf found!")
            continue
         
         #MSE
         mse.append(((imp_emb - real_emb)**2).mean())
         # Cosine similarity
         numerator = np.dot(real_emb, imp_emb)
         denominator = (np.linalg.norm(real_emb) * np.linalg.norm(imp_emb) + eps)
         cosine_sim.append(numerator / denominator)

      return np.mean(mse), np.mean(cosine_sim)

    # Function to compute reconstruction errors for a modality
    def compute_reconstruction_errors_for_modality(self, input_path, result_path, missing_items_npy):  
        
        real_embeddings = [np.load(os.path.join(input_path, file)) for file in os.listdir(input_path) 
                           if file.endswith('.npy') and file in missing_items_npy]
        imputed_embeddings = [np.load(os.path.join(result_path, file)) for file in os.listdir(result_path) 
                              if file.endswith('.npy') and file in missing_items_npy]

        return self.compute_reconstruction_errors(real_embeddings, imputed_embeddings)


    ### Cleanup function
    def cleanup_temp_resources(self, paths):
        for path in paths:
            if path is None:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


    # Imputation method implemented by each child class 
    def impute(self):
        raise NotImplementedError("Implement the impute() method in the child class.")