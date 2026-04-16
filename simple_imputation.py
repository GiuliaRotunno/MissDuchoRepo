import numpy as np
import os
from loguru import logger
import glob 
import pandas as pd

def compute_embedding_dim(input_path):
      embedding_files = glob.glob(os.path.join(input_path, '*.npy'))
      # Per inferire le dimensioni, usa il primo file trovato di qualsiasi modello
      if embedding_files:
         sample_embedding = np.load(embedding_files[0])
         embedding_dim = sample_embedding.shape[1]
         logger.info(f"Inferred embedding dimension: {embedding_dim}")
      else:
         raise ValueError("Could not infer embedding dimension. No .npy files found for any model.")
      return embedding_dim

def compute_missing_items(dataset_path, missing_file):
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

def compute_artificial_missing_items(dataset_path, missing_percentage, items_file, seed, file_name):

   if missing_percentage < 0.0 or missing_percentage > 0.99:
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
      
     artificial_missing_file = f'local/data/demo_giulia/{file_name}.tsv'  #crea missing_visual_file per quel ratio
     with open(artificial_missing_file, "w") as f:
        for i, item in enumerate(items):
            if mask[i]:
                f.write(f"{item}\n")

     with open(artificial_missing_file, 'r') as f:
         artificial_missing_data = f.readlines()
         artificial_missing_items = [line.strip() for line in artificial_missing_data]
         
   return artificial_missing_items

###funzioni per errore di ricostruzione
   
def compute_reconstruction_errors(real_embeddings, imputed_embeddings, eps=1e-8):
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

      MSE = np.mean(mse) 
      COSINE_SIM = np.mean(cosine_sim)
      return MSE, COSINE_SIM







'''
def compute_missing_items(dataset_path, missing_file):
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
'''

'''old

def compute_embedding_dim(self):   #se model_path è una lista?
      embedding_files = glob.glob(os.path.join(self.model_path, '*.npy'))
      # Per inferire le dimensioni, usa il primo file trovato di qualsiasi modello
      if embedding_files:
         sample_embedding = np.load(embedding_files[0])
         embedding_dim = sample_embedding.shape[1]
         logger.info(f"Inferred embedding dimension: {embedding_dim}")
      else:
         raise ValueError("Could not infer embedding dimension. No .npy files found for any model.")
      return embedding_dim

def compute_imputed_path(self):
      # full_output_path = os.path.join(dataset_path, output_path) #crea la cartella di output
      parent_dir = os.path.dirname('demo_giulia')
      imputed_path = os.path.join(parent_dir, "imputed", self.model_path) #crea la cartella di output
      os.makedirs(imputed_path, exist_ok=True)
      return imputed_path
   
def compute_missing_items(self):
       missing_file_path = os.path.join(self.dataset_path, self.missing_file)
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

'''