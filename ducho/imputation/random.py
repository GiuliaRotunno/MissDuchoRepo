import numpy as np
import os
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

#classe figlia di ImputatorFather
class RandomImputator(ImputatorFather):  #Apply random imputation for items with missing modalities
   
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
      embedding_dim = self.compute_embedding_dim(self.input_path)  #funzione in simple_imputation
      np.random.seed(self.seed) #seed per la riproducibilità

      result_path = os.path.join(self.output_path, "random")
      os.makedirs(result_path, exist_ok=True)

      if self.missing_percentage is not None:
         errors = {}
         for percentage in self.missing_percentage:
            missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_random")  #funzione in simple_imputation
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            for item_id in missing_items:
                 random_embedding = np.random.randn(embedding_dim).astype(np.float32)
                 np.save(os.path.join(result_path_percentage, f"{item_id}.npy"), random_embedding)
            
            # Calcola errore di ricostruzione
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
            errors[percentage] = (mse, cosine_sim)
           
         return errors
      
      else:
           missing_items = self.compute_missing_items(self.dataset_path, self.missing_file) #funzione in simple_imputation
           for item_id in missing_items:
               random_embedding = np.random.randn(embedding_dim).astype(np.float32)
               np.save(os.path.join(result_path, f"{item_id}.npy"), random_embedding)


'''old
import numpy as np
import os
from simple_imputation import compute_embedding_dim, compute_imputed_path, compute_missing_items

class RandomImputator():
   """
      Apply random imputation for items with missing modalities.
      Args: missing_items (List[str]): List IDs of item with missing modalities 
            full_output_path (str): Path to save imputed embeddings.
            embedding_dim (int): Dimension of embeddings.
            seed (int): Random seed for reproducibility.
    """
   def __init__(self, dataset_path, missing_file, model_path, seed):
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.model_path = model_path
        self.seed = seed
   
   def impute(self):
      embedding_dim = compute_embedding_dim(self)
      np.random.seed(self.seed) #seed per la riproducibilità

      imputed_path = compute_imputed_path(self)
      result_path = os.path.join(imputed_path, "random")
      os.makedirs(result_path, exist_ok=True)

      missing_items = compute_missing_items(self)
      for item_id in missing_items:
         random_embedding = np.random.randn(embedding_dim).astype(np.float32)
         np.save(os.path.join(result_path, f"{item_id}.npy"), random_embedding)

'''





''' old implementation
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

class RandomImputator(ImputatorFather):
   
   def __init__(self, list_of_modalities, random_seed=42):     #problema random seed 
      self.random_seed = random_seed
      super().__init__(list_of_modalities)
  
   def impute(self):
      rng = np.random.RandomState(self.random_seed)
      list_of_modalities = self.list_of_modalities   # list_of_modalities = ['visual', 'textual']

      if 'visual' in list_of_modalities:
         missing_visual, visual_folder, visual_shape = self.read_missings(list_of_modalities=['visual'])
         output_visual = f'local/data/demo_baby/visual_embeddings_random'
         if not os.path.exists(output_visual):
            os.makedirs(output_visual)
         for miss in missing_visual:
             np.save(os.path.join(output_visual, f'{miss}.npy'), rng.rand(*visual_shape))
      
      if 'textual' in list_of_modalities:
         missing_textual, textual_folder, textual_shape = self.read_missings(list_of_modalities=['textual'])
         output_textual = f'local/data/demo_baby/textual_embeddings_random'  
         if not os.path.exists(output_textual):
            os.makedirs(output_textual)
         for miss in missing_textual:
             np.save(os.path.join(output_textual, f'{miss}.npy'), rng.rand(*textual_shape))
'''
      

    

