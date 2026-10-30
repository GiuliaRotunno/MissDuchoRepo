import numpy as np
import os
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# RANDOM: apply random imputation for missing items by generating random embeddings with values sampled from a uniform distribution.

# Child class of ImputatorFather
class RandomImputator(ImputatorFather):
   
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
      np.random.seed(self.seed)

      result_path = os.path.join(self.output_path, "random")
      os.makedirs(result_path, exist_ok=True)

      if self.missing_percentage is not None:
         errors = {}
         for percentage in self.missing_percentage:
            missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_random")
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            for item_id in missing_items:
                 random_embedding = np.random.randn(embedding_dim).astype(np.float32)
                 np.save(os.path.join(result_path_percentage, f"{item_id}.npy"), random_embedding)
            
            # Compute reconstruction error
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
            errors[percentage] = (mse, cosine_sim)
           
         return errors
      
      else: 
           missing_items = self.compute_missing_items(self.dataset_path, self.missing_file)
           for item_id in missing_items:
               random_embedding = np.random.randn(embedding_dim).astype(np.float32)
               np.save(os.path.join(result_path, f"{item_id}.npy"), random_embedding)