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


    ### funzioni per simple_imputation
    def compute_embedding_dim(self, input_path):
      embedding_files = glob.glob(os.path.join(input_path, '*.npy'))
      # Per inferire le dimensioni, usa il primo file trovato di qualsiasi modello
      if embedding_files:
         sample_embedding = np.load(embedding_files[0])
         embedding_dim = sample_embedding.shape[1]
         logger.info(f"Inferred embedding dimension: {embedding_dim}")
      else:
         raise ValueError("Could not infer embedding dimension. No .npy files found for any model.")
      return embedding_dim
    
    def compute_missing_items(self, dataset_path, missing_file):
         missing_file_path = os.path.join(dataset_path, missing_file)
         if not os.path.exists(missing_file_path):
            logger.error(f"Missing file not found: {missing_file_path}")
         logger.info(f"Applying imputation using missing data from {missing_file_path}")

         try: #il missing file è un file testuale con un item per riga
              with open(missing_file_path, 'r') as f:
                    missing_data = f.readlines()
                    missing_items = [line.strip() for line in missing_data]
         except Exception as e:
              logger.error(f"Error loading missing data file: {str(e)}")
         return missing_items
    
    def compute_artificial_missing_items(self, dataset_path, missing_percentage, items_file, seed, file_name):

        if missing_percentage < 0.0 or missing_percentage > 1.1:
            logger.error(f"Percentage of missing items must be between 0.0 and 0.99")
        else:
            items_file_path = os.path.join(dataset_path, items_file)
            if not os.path.exists(items_file_path):
                    logger.error(f"Items file not found: {items_file_path}")
            logger.info(f"Computing artificial missing items")
            df = pd.read_csv(items_file_path, sep="\t")            #lettura file tsv di item con tutte le modalità presenti
            items = df["item_name"].tolist()                       #lista item
        
            np.random.seed(seed) #seed per la riproducibilità
            mask  = np.random.rand(len(items)) < missing_percentage #crea mask visual per gli item
            if not mask.any():   
                mask[np.random.randint(len(items))] = True  # Garantisce almeno un masked per modalità
            
            artificial_missing_file = f'{dataset_path}/{file_name}.tsv'  #crea missing_visual_file per quel ratio
            with open(artificial_missing_file, "w") as f:
                for i, item in enumerate(items):
                    if mask[i]:
                        f.write(f"{item}\n")

            with open(artificial_missing_file, 'r') as f:
                artificial_missing_data = f.readlines()
                artificial_missing_items = [line.strip() for line in artificial_missing_data]
                
        return artificial_missing_items
 
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
                
                # carica gli item che hanno effettivamente un embedding
                items_with_embeddings = set([f.split('.npy')[0] for f in os.listdir(self.input_path)])
                # per ogni item, trova i suoi vicini nel train e verifica che almeno uno abbia embedding
                item_neighbors = train.groupby(1)[0].apply(set).to_dict()  # item -> set di utenti
                user_items = train.groupby(0)[1].apply(set).to_dict()      # utente -> set di item
                
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

    ### funzioni per graph_imputation
    def get_item_item(self, train, num_items):  # compute item_item matrix
       user_item = sp.coo_matrix(([1.0] * len(train), (train[0].tolist(), train[1].tolist())), shape=(train[0].nunique(), num_items), dtype=np.float32)
       item_item = user_item.transpose().dot(user_item).toarray()
       knn_val, knn_ind = torch.topk(torch.tensor(item_item, device=self.device), self.top_k, dim=-1)
       items_cols = torch.flatten(knn_ind).to(self.device)
       ir = torch.tensor(list(range(item_item.shape[0])), dtype=torch.int64, device=self.device)
       items_rows = torch.repeat_interleave(ir, self.top_k).to(self.device)
       final_adj = SparseTensor(row=items_rows, col=items_cols, value=torch.tensor([1.0] * items_rows.shape[0], device=self.device), sparse_sizes=(item_item.shape[0], item_item.shape[0]))
       return final_adj
   
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
   
    def inverse_mapping(self,items_map, item_mapped):
        inverse_map = {v: k for k, v in items_map.items()}
        item_original = inverse_map.get(item_mapped)
        return item_original

    def create_non_masked_folder(self, input_path, missing_items_npy, folder_path): #Crea un folder con gli item non mascherati
   
        os.makedirs(folder_path, exist_ok=True)
        for file in os.listdir(input_path):
            if file.endswith('.npy') and (file not in missing_items_npy): 
                np.save(f'{folder_path}/{file}', np.load(os.path.join(input_path, file)))
    

    ### funzioni per autoencoder imputation
    def set_seed(self): #settare seed per riproducibilità
      
       random.seed(self.seed)
       np.random.seed(self.seed)
       torch.manual_seed(self.seed)
       torch.cuda.manual_seed_all(self.seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

    def normalize_features(self, visual_features, textual_features): #Normalizza le features e calcola le medie globali
       
        # Normalizzazione visual
        visual_tensor = torch.from_numpy(visual_features)
        visual_min = visual_tensor.min(0, keepdim=True)[0]
        visual_max = visual_tensor.max(0, keepdim=True)[0]
        visual_normalized = (visual_tensor - visual_min) / (visual_max - visual_min + 1e-8)
        visual_normalized = visual_normalized.to(self.device)
        
        # Normalizzazione textual
        textual_tensor = torch.from_numpy(textual_features)
        textual_min = textual_tensor.min(0, keepdim=True)[0]
        textual_max = textual_tensor.max(0, keepdim=True)[0]
        textual_normalized = (textual_tensor - textual_min) / (textual_max - textual_min + 1e-8)
        textual_normalized = textual_normalized.to(self.device)
        
        # Sposta min/max su device
        visual_min = visual_min.to(self.device)
        visual_max = visual_max.to(self.device)
        textual_min = textual_min.to(self.device)
        textual_max = textual_max.to(self.device)
        
        # Calcola medie globali come fallback
        visual_global_mean = visual_tensor.mean(dim=0).to(self.device)
        textual_global_mean = textual_tensor.mean(dim=0).to(self.device)
        
        return (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean)
 
    def load_features(self, visual_folder, textual_folder):  #Carica le features visual e textual e trova gli item con entrambe le modalità
        
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
  
    def load_indexed_features(self, visual_embeddings_folder_indexed, textual_embeddings_folder_indexed, missing_visual_indexed, missing_textual_indexed):  #Carica le features indicizzate e trova gli item con entrambe le modalità
        
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

    def prepare_adjacency_matrix(self, train_indexed_path, all_present):  #Prepara la matrice di adiacenza dal training set
        
        train = pd.read_csv(train_indexed_path, sep='\t', header=None)
        train = train[train[1].isin(all_present)]
        inner_map_user = {u: idx for idx, u in enumerate(train[0].unique())}
        inner_map_item = {item: idx for idx, item in enumerate(all_present)}
        train[0] = train[0].map(inner_map_user)
        train[1] = train[1].map(inner_map_item)
        adj = self.get_item_item(train, num_items=len(all_present))
        return adj


    ### funzioni per deep-learning imputation
    def load_data(self, visual_folder, textual_folder): #Carica i dati e identifica items con entrambe le modalità

        # Identifica items con entrambe le modalità
        visual_items = set(os.listdir(visual_folder))
        textual_items = set(os.listdir(textual_folder))
        all_present = sorted(list(visual_items & textual_items)) #all_present: list di filenames con entrambe le modalità
        
        # Carica features per items completi
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
        visual_features = np.array(visual_features, dtype=np.float32) #visual_features: numpy array [N, visual_dim]
        textual_features = np.array(textual_features, dtype=np.float32) #textual_features: numpy array [N, textual_dim]
        
        return visual_features, textual_features, all_present
     
    def normalize_data(self, visual_features, textual_features): #Normalizza features in [0, 1]
        visual_tensor = torch.from_numpy(visual_features).float()
        textual_tensor = torch.from_numpy(textual_features).float()
        # Calcola min/max
        self.visual_min = visual_tensor.min(dim=0, keepdim=True)[0]
        self.visual_max = visual_tensor.max(dim=0, keepdim=True)[0]
        self.textual_min = textual_tensor.min(dim=0, keepdim=True)[0]
        self.textual_max = textual_tensor.max(dim=0, keepdim=True)[0]
        # Calcola medie globali come fallback per item con entrambe le modalità mancanti
        self.visual_global_mean = visual_tensor.mean(dim=0)
        self.textual_global_mean = textual_tensor.mean(dim=0)
        # Normalizza
        visual_norm = (visual_tensor - self.visual_min) / (self.visual_max - self.visual_min + 1e-8)
        textual_norm = (textual_tensor - self.textual_min) / (self.textual_max - self.textual_min + 1e-8)
        # Sposta su device
        visual_norm = visual_norm.to(self.device)
        textual_norm = textual_norm.to(self.device)
        self.visual_min = self.visual_min.to(self.device)
        self.visual_max = self.visual_max.to(self.device)
        self.textual_min = self.textual_min.to(self.device)
        self.textual_max = self.textual_max.to(self.device)
        self.visual_global_mean = self.visual_global_mean.to(self.device)
        self.textual_global_mean = self.textual_global_mean.to(self.device)
        return visual_norm, textual_norm

    def denormalize(self, data, modality='visual'):  #Denormalizza da [0,1] al range originale
        if modality == 'visual':
            return data * (self.visual_max - self.visual_min) + self.visual_min
        else: #modality == 'textual'
            return data * (self.textual_max - self.textual_min) + self.textual_min
    

    ### funzioni per il calcolo degli errori di ricostruzione
    def compute_reconstruction_errors(self, real_embeddings, imputed_embeddings, eps=1e-8):
      mse = []
      cosine_sim = []

      for real_emb, imp_emb in zip(real_embeddings, imputed_embeddings):
         
         real_emb = np.asarray(real_emb).reshape(-1)
         imp_emb = np.asarray(imp_emb).reshape(-1)

         #MSE error
         mse.append(((imp_emb - real_emb)**2).mean())
         
         # Cosine similarity
         numerator = np.dot(real_emb, imp_emb)
         denominator = (np.linalg.norm(real_emb) * np.linalg.norm(imp_emb) + eps)
         cosine_sim.append(numerator / denominator)

      return np.mean(mse),np.mean(cosine_sim)

    def compute_reconstruction_errors_for_modality(self, input_path, result_path, missing_items_npy):  
        
        real_embeddings = [np.load(os.path.join(input_path, file)) for file in os.listdir(input_path) if file.endswith('.npy') and file in missing_items_npy]
        imputed_embeddings = [np.load(os.path.join(result_path, file)) for file in os.listdir(result_path) if file.endswith('.npy') and file in missing_items_npy]
        return self.compute_reconstruction_errors(real_embeddings, imputed_embeddings)


    ### funzione di cleanup
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


    def impute(self):
        raise NotImplementedError("Implementare impute() nella classe figlia")





'''
import pandas as pd 
import numpy as np
import os
import torch
import tensorflow as tf

class ImputatorFather:
    """
    Represents a imputator object.

    This class provides functionality for feature imputation using various methods.

    Attributes:
        _method: The method for imputation.
        _method_name: The name of the method.
        _gpu (str): The GPU index or '-1' for CPU.
        _device: The device for computation (GPU, MPS, CPU).
    """
    def __init__(self, gpu='-1'):
        """
        Initialize the Imputator object.

        Args:
            gpu (str, optional): The GPU index or '-1' for CPU.

        Returns:
            None
        """
        self._method = None
        self._method_name = None
        self._gpu = gpu
        self._device = torch.device(f'cuda:{self._gpu}' if torch.cuda.is_available() else 'cpu')
        self._device = torch.device(f'mps' if torch.backends.mps.is_available() else self._device)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def set_method(self, method_name):
        pass

    def impute(self, embedding):
        pass

'''

'''
# find the items with missing modalities (total items - extracted), take the extracted features, impute with methods
class ImputatorFather:
    """ Repesent an imputator object. This class provides functionality for missing feature imputation using various methods.

    Attributes:
        _list_of_modalities (list): list containing the name of the modalities to impute.
        _items_missing_features (dict): dictionary containing the missing modalities and the corresponding items (missing_modality, item).
        _extracted_features (dict): dictionary containing the modalities and the extracted features
    """ 
   
    def __init__(self, list_of_modalities):
       self.list_of_modalities = list_of_modalities
       # self.items_missing_features = items_missing_features
       # self.extracted_features = extracted_features

    def read_missings(self, list_of_modalities):
        #self.list_of_modalities = list_of_modalities   # list_of_modalities = ['visual', 'textual']

        if 'visual' in list_of_modalities:
         try:
             missing_visual = pd.read_csv(os.path.join(f'local/data/demo_baby', 'missing_visual.tsv'), sep='\t', header=None)
             missing_visual = set(missing_visual[0].tolist())
         except pd.errors.EmptyDataError:
            missing_visual = set()
         visual_folder = f'local/data/demo_baby/visual_embeddings/torch/ResNet50/avgpool'
         visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
         return missing_visual, visual_folder, visual_shape

        if 'textual' in list_of_modalities: 
         try:
             missing_textual = pd.read_csv(os.path.join(f'local/data/demo_baby', 'missing_textual.tsv'), sep='\t', header=None)
             missing_textual = set(missing_textual[0].tolist())
         except pd.errors.EmptyDataError:
            missing_textual = set()
         textual_folder = f'local/data/demo_baby/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1' 
         textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape 
         return missing_textual, textual_folder, textual_shape
        
    def load_available_features(self, list_of_modalities, visual_folder, visual_shape, textual_folder, textual_shape): #estesa in tutte le classi, in init, no zeros and random methods
        #list_of_modalities = self.list_of_modalities   # list_of_modalities = ['visual', 'textual']
        
        if 'visual' in list_of_modalities:
            num_items_visual = len(os.listdir(visual_folder))
            visual_features = np.empty((num_items_visual, visual_shape[-1])) if num_items_visual else None
            if visual_features is not None:
               visual_items = os.listdir(visual_folder)
               for idx, it in enumerate(visual_items):
                   visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
            return visual_features

        if 'textual' in list_of_modalities:
            num_items_textual = len(os.listdir(textual_folder))
            textual_features = np.empty((num_items_textual, textual_shape[-1])) if num_items_textual else None
            if textual_features is not None:
               textual_items = os.listdir(textual_folder)
               for idx, it in enumerate(textual_items):
                   textual_features[idx, :] = np.load(os.path.join(textual_folder, it))
            return textual_features

    def impute(self, list_of_modalities):
        pass


   # items_missing_features = ['visual': [0,1,2], 'textual': [0,1,2]]
   # output_visual = list of (item, imputated visual embedding)
   # output_textual = list of (item, imputated textual embedding)
   # dict(modality, item available) #not for every method, not in imputator father, in method classes
    """_items_available_features (dict): dictionary containing the available modalities and the corresponding items (available_modality, item)."""
'''