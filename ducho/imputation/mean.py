import numpy as np
import os
from loguru import logger
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# MEAN: apply mean imputation for missing items by computing the mean of existing embeddings.

# Child class of ImputatorFather
class MeanImputator(ImputatorFather): 
        
    def __init__(self, dataset_path, missing_file, input_paths, output_paths, missing_percentage, items_file, seed):
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.input_path = input_paths
        self.output_path = output_paths
        self.missing_percentage = missing_percentage
        self.items_file = items_file
        self.seed = seed
    
    def impute(self):
        embedding_dim = self.compute_embedding_dim(self.input_path)
        
        result_path = os.path.join(self.output_path, "mean")
        os.makedirs(result_path, exist_ok=True)

        if self.missing_percentage is not None:
            errors = {}
            for percentage in self.missing_percentage:
              missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_mean")
              result_path_percentage = os.path.join(result_path, f"{percentage}")
              os.makedirs(result_path_percentage, exist_ok=True)
              missing_items_npy = [f'{item}.npy' for item in missing_items]
              # Calculate average on not-masked
              existing_embeddings = []
              if os.path.exists(self.input_path):
                for file in os.listdir(self.input_path):
                   if file.endswith('.npy') and (file not in missing_items_npy):
                      try:
                          embedding = np.load(os.path.join(self.input_path, file))
                          if embedding.shape[1] == embedding_dim:
                             existing_embeddings.append(embedding)
                      except Exception as e:
                          os.logger.warning(f"Error loading embedding {file}: {str(e)}")
              # Compute average of existing embeddings
              if existing_embeddings: 
                 mean_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
                 logger.info(f"Computed mean embedding from {len(existing_embeddings)} existing embeddings")
              else:
                 logger.info("No existing embeddings found. Using zeros instead.")
                 mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
              for item_id in missing_items:
                  np.save(os.path.join(result_path_percentage, f"{item_id}.npy"), mean_embedding)

              # Compute reconstruction error
              mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
              errors[percentage] = (mse, cosine_sim)
            
            return errors 

        else:
           missing_items = self.compute_missing_items(self.dataset_path, self.missing_file) 
           existing_embeddings = []
           if os.path.exists(self.input_path):
             for file in os.listdir(self.input_path):
                if file.endswith('.npy'):
                    try:
                        embedding = np.load(os.path.join(self.input_path, file))
                        if embedding.shape[1] == embedding_dim:
                           existing_embeddings.append(embedding)
                    except Exception as e:
                        os.logger.warning(f"Error loading embedding {file}: {str(e)}")  
           # Compute average over all available embeddings                        
           if existing_embeddings:
              mean_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
              logger.info(f"Computed mean embedding from {len(existing_embeddings)} existing embeddings")
           else:
              logger.info("No existing embeddings found. Using zeros instead.")
              mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
           for item_id in missing_items:
               np.save(os.path.join(result_path, f"{item_id}.npy"), mean_embedding)