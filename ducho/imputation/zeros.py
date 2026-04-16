import numpy as np 
import os
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

#classe figlia di ImputatorFather
class ZerosImputator(ImputatorFather):  #Apply zeros imputation for missing items   
           
   def __init__(self, dataset_path, missing_file, input_paths, output_paths, missing_percentage, items_file, seed):
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.input_path = input_paths
        self.output_path = output_paths
        self.missing_percentage = missing_percentage #lista
        self.items_file = items_file
        self.seed = seed

   def impute(self):
      embedding_dim = self.compute_embedding_dim(self.input_path)  #funzione in simple_imputation
      zeros_embedding = np.zeros(embedding_dim, dtype=np.float32)
        
      result_path = os.path.join(self.output_path, "zeros")
      os.makedirs(result_path, exist_ok=True)
        
      if self.missing_percentage is not None:
         errors = {}
         for percentage in self.missing_percentage:
            missing_items = self.compute_artificial_missing_items(self.dataset_path, percentage, self.items_file, self.seed, file_name=f"artificial_missing_file_{percentage}_zeros")  #funzione in simple_imputation 
            result_path_percentage = os.path.join(result_path, f"{percentage}")
            os.makedirs(result_path_percentage, exist_ok=True)
            for item_id in missing_items:  
                 np.save(os.path.join(result_path_percentage, f"{item_id}.npy"), zeros_embedding)
            
            # Calcola errore di ricostruzione
            missing_items_npy = [f'{item}.npy' for item in missing_items]
            mse, cosine_sim = self.compute_reconstruction_errors_for_modality(self.input_path, result_path_percentage, missing_items_npy)
            errors[percentage] = (mse, cosine_sim)
         
         return errors
      
      else:
         missing_items = self.compute_missing_items(self.dataset_path, self.missing_file) #funzione in simple_imputation
         for item_id in missing_items:  
             np.save(os.path.join(result_path, f"{item_id}.npy"), zeros_embedding)

      
       
             
         
   
'''old
   def impute(self):
      embedding_dim = compute_embedding_dim(self.input_path)  #funzione in simple_imputation
      zeros_embedding = np.zeros(embedding_dim, dtype=np.float32)
        
      result_path = os.path.join(self.output_path, "zeros")
      os.makedirs(result_path, exist_ok=True)
        
      if self.missing_percentage is not None:
         #for percentage in self.missing_percentage:
            missing_items = compute_artificial_missing_items(self.dataset_path, self.missing_percentage, self.items_file, self.seed, file_name='artificial_missing_file')  #funzione in simple_imputation 
      else:
         missing_items = compute_missing_items(self.dataset_path, self.missing_file) #funzione in simple_imputation
      
      for item_id in missing_items:  
         np.save(os.path.join(result_path, f"{item_id}.npy"), zeros_embedding)


      if self.missing_percentage is not None: #per l'errore di ricostruzione
         missing_items_npy = []
         for item in missing_items:
            missing_items_npy.append(f'{item}.npy')
         real_embeddings = []
         for file in os.listdir(self.input_path):
            if file.endswith('.npy') and (file in missing_items_npy):
                real_embeddings.append(np.load(os.path.join(self.input_path, file)))

         imputed_embeddings = []
         for file in os.listdir(result_path):
            if file.endswith('.npy') and (file in missing_items_npy):
                imputed_embeddings.append(np.load(os.path.join(result_path, file)))
         
         mse_error = compute_reconstruction_errors(real_embeddings, imputed_embeddings)
         return mse_error
'''

''' old
import numpy as np
import os
from simple_imputation import compute_embedding_dim, compute_imputed_path, compute_missing_items

class ZerosImputator(): 
   """
      Apply zeros imputation for missing items.   
      Args: missing_items (List[str]): List of missing item IDs.
            full_output_path (str): Path to save imputed embeddings.
            embedding_dim (int): Dimension of embedding vectors.
   """
   def __init__(self, dataset_path, missing_file, model_path):
        self.dataset_path = dataset_path
        self.missing_file = missing_file
        self.model_path = model_path

   def impute(self):
        embedding_dim = compute_embedding_dim(self)
        zeros_embedding = np.zeros(embedding_dim, dtype=np.float32)

        imputed_path = compute_imputed_path(self)
        result_path = os.path.join(imputed_path, "zeros")
        os.makedirs(result_path, exist_ok=True)
        
        missing_items = compute_missing_items(self)
        for item_id in missing_items:  
            np.save(os.path.join(result_path, f"{item_id}.npy"), zeros_embedding)

'''

 
''' old implementation
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

class ZerosImputator(ImputatorFather): 

   def __init__(self, list_of_modalities):
      super().__init__(list_of_modalities)

   def impute(self): 
      list_of_modalities = self.list_of_modalities   # list_of_modalities = ['visual', 'textual']

      if 'visual' in list_of_modalities:
         missing_visual, visual_folder, visual_shape = self.read_missings(list_of_modalities=['visual'])
         output_visual = f'local/data/demo_baby/visual_embeddings_zeros'
         if not os.path.exists(output_visual): 
            os.makedirs(output_visual)
         for miss in missing_visual:
             np.save(os.path.join(output_visual, f'{miss}.npy'), np.zeros(visual_shape, dtype=np.float32))
      
      if 'textual' in list_of_modalities:
         missing_textual, textual_folder, textual_shape = self.read_missings(list_of_modalities=['textual'])
         output_textual = f'local/data/demo_baby/textual_embeddings_zeros'  
         if not os.path.exists(output_textual):
            os.makedirs(output_textual)
         for miss in missing_textual:
             np.save(os.path.join(output_textual, f'{miss}.npy'), np.zeros(textual_shape, dtype=np.float32))

'''


   
