import numpy as np
import os
from loguru import logger
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

#classe figlia di ImputatorFather
class MeanImputator(ImputatorFather): #Apply mean imputation for missing items by computing the mean of existing embeddings
       
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
        
        result_path = os.path.join(self.output_path, "mean")
        os.makedirs(result_path, exist_ok=True)

        if self.missing_percentage is not None:
            errors = {}
            for percentage in self.missing_percentage:
              missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_mean")  #funzione in simple_imputation
              result_path_percentage = os.path.join(result_path, f"{percentage}")
              os.makedirs(result_path_percentage, exist_ok=True)
              missing_items_npy = [f'{item}.npy' for item in missing_items]
              #calcolare media su non masked
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
              if existing_embeddings: #calcolare media degli embedding esistenti
                 mean_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
                 logger.info(f"Computed mean embedding from {len(existing_embeddings)} existing embeddings")
              else:
                 logger.info("No existing embeddings found. Using zeros instead.")
                 mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
              for item_id in missing_items:
                  np.save(os.path.join(result_path_percentage, f"{item_id}.npy"), mean_embedding)

              #calcolo errore di ricostruzione
              mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
              errors[percentage] = (mse, cosine_sim)
            
            return errors

        else:
           missing_items = self.compute_missing_items(self.dataset_path, self.missing_file) #funzione in simple_imputation 
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
           if existing_embeddings: #calcolare media su tutti gli embedding disponibili
              mean_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
              logger.info(f"Computed mean embedding from {len(existing_embeddings)} existing embeddings")
           else:
              logger.info("No existing embeddings found. Using zeros instead.")
              mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
           for item_id in missing_items:
               np.save(os.path.join(result_path, f"{item_id}.npy"), mean_embedding)




'''old
import numpy as np
import os
from simple_imputation import compute_embedding_dim, compute_imputed_path, compute_missing_items

class MeanImputator():
    """
       Apply mean imputation for missing items by computing the mean of existing embeddings.
       Args: missing_items (List[str]): List of missing item IDs.
             full_output_path (str): Path to save imputed embeddings.
             embedding_dim (int): Dimension of embeddings.
             input_path (str): Path to input data containing existing embeddings.
    """ 
    def __init__(self, dataset_path, missing_file, model_path):
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.model_path = model_path
    
    def impute(self):
        embedding_dim = compute_embedding_dim(self)
        existing_embeddings = []
        if os.path.exists(self.model_path):
            for file in os.listdir(self.model_path):
                if file.endswith('.npy'):
                    try:
                        embedding = np.load(os.path.join(self.model_path, file))
                        if embedding.shape[1] == embedding_dim:
                           existing_embeddings.append(embedding)
                    except Exception as e:
                        os.logger.warning(f"Error loading embedding {file}: {str(e)}")
    
        if existing_embeddings: #calcola la media degli embedding esistenti
           mean_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
           print(f"Computed mean embedding from {len(existing_embeddings)} existing embeddings")
        else:
           print("No existing embeddings found. Using zeros instead.")
           mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
        
        imputed_path = compute_imputed_path(self)       
        result_path = os.path.join(imputed_path, "mean")
        os.makedirs(result_path, exist_ok=True)
        
        missing_items = compute_missing_items(self)
        for item_id in missing_items:
            np.save(os.path.join(result_path, f"{item_id}.npy"), mean_embedding)

'''



''' old implementation
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

class MeanImputator(ImputatorFather):
    
    def __init__(self, list_of_modalities):
      super().__init__(list_of_modalities)

    def impute(self):
        list_of_modalities = self.list_of_modalities   # list_of_modalities = ['visual', 'textual']
        
        if 'visual' in list_of_modalities:
            missing_visual, visual_folder, visual_shape = self.read_missings(list_of_modalities=['visual'])
            output_visual = f'local/data/demo_baby/visual_embeddings_mean'
            if not os.path.exists(output_visual):
               os.makedirs(output_visual)
            visual_features = self.load_available_features(list_of_modalities=['visual'], visual_folder=visual_folder, visual_shape=visual_shape, textual_folder=None, textual_shape=None)
            mean_visual = visual_features.mean(axis=0, keepdims=True)
            for miss in missing_visual:
                np.save(os.path.join(output_visual, f'{miss}.npy'), mean_visual)      


        if 'textual' in list_of_modalities:
            missing_textual, textual_folder, textual_shape = self.read_missings(list_of_modalities=['textual'])
            output_textual = f'local/data/demo_baby/textual_embeddings_mean'
            if not os.path.exists(output_textual):
               os.makedirs(output_textual)
            textual_features = self.load_available_features(list_of_modalities=['textual'], visual_folder=None, visual_shape=None, textual_folder=textual_folder, textual_shape=textual_shape)                         
            mean_textual = textual_features.mean(axis=0, keepdims=True)
            for miss in missing_textual:
                np.save(os.path.join(output_textual, f'{miss}.npy'), mean_textual)
'''

   