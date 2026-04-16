import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ducho.imputation.dicmor_imputation.dicmormodel import DICMORModel
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

#classe figlia di ImputatorFather
class DicmorImputator(ImputatorFather): 
    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, embd_size_v, embd_size_t, gpu_ids, seed, missing_percentage_visual, missing_percentage_textual, items_file, 
                num_epochs, learning_rate, batch_size, log_p_weight, n_flow, n_block, rcan_blocks):
        self.dataset_path = dataset_path
        self.missing_visual_file = missing_visual_file
        self.missing_textual_file = missing_textual_file
        self.input_paths = input_paths
        self.output_paths = output_paths
        
        # Assicura che le dimensioni siano int
        if isinstance(visual_dim, (tuple, list)):
            visual_dim = visual_dim[0]
        if isinstance(textual_dim, (tuple, list)):
            textual_dim = textual_dim[0]
        self.visual_dim = int(visual_dim)
        self.textual_dim = int(textual_dim)
        self.embd_size_v = embd_size_v      
        self.embd_size_t = embd_size_t
        self.gpu_ids = gpu_ids
        self.seed = seed
        self.missing_percentage_visual = missing_percentage_visual
        self.missing_percentage_textual = missing_percentage_textual
        self.items_file = items_file
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_p_weight = log_p_weight
        self.n_flow = n_flow
        self.n_block = n_block
        self.rcan_blocks = rcan_blocks
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')

        # Statistiche per normalizzazione
        self.visual_min = None
        self.visual_max = None
        self.textual_min = None
        self.textual_max = None
    
    #denormalize?

    def train_dicmor(self, visual_features, textual_features, num_epochs, batch_size, lr, log_p_weight):   #dicmor num_epochs=8, batch_size=16, lr=0.0001
        # Normalizza
        visual_norm, textual_norm = self.normalize_data(visual_features, textual_features)
        # Setup optimizers
        self.model.setup_optimizers(lr=lr)
        # Crea dataloader
        dataset = TensorDataset(visual_norm, textual_norm)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss_total = 0
            epoch_loss_rec = 0
            num_batches = 0
            for visual_batch, textual_batch in dataloader:
                # Simula missing modality (alterna tra textual e visual missing)
                if num_batches % 2 == 0:
                    num_modal = 1  # Simula una modalità disponibile
                else:
                    num_modal = 2  # Entrambe disponibili
                # Optimize
                loss_dict = self.model.optimize_parameters(visual_batch, textual_batch, num_modal, log_p_weight=log_p_weight)
                epoch_loss_total += loss_dict['loss_total']
                epoch_loss_rec += loss_dict['loss_rec']
                num_batches += 1
            avg_loss_total = epoch_loss_total / num_batches
            avg_loss_rec = epoch_loss_rec / num_batches
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {avg_loss_total:.4f}, Rec Loss: {avg_loss_rec:.4f}")
    
    def impute_missing(self, visual_folder, textual_folder, missing_visual, missing_textual, output_visual_folder, output_textual_folder):
        #Imputa le modalità mancanti usando il modello trainato, imputa in embedding space (128 dim), non nello spazio originale        
        os.makedirs(output_visual_folder, exist_ok=True)
        os.makedirs(output_textual_folder, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            # Imputa missing textual da visual
            for item in tqdm(missing_textual, desc="Textual imputation"):
                if item in missing_visual:
                    # Usa la media globale invece di zeri
                    textual_imputed = self.textual_global_mean.cpu().numpy()
                    #textual_imputed = np.zeros(self.textual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica visual e imputa textual
                    visual_path = os.path.join(visual_folder, item + ".npy")
                    visual_feat = np.load(visual_path)
                    visual_tensor = torch.from_numpy(visual_feat).float()
                    # Normalizza
                    visual_norm = (visual_tensor - self.visual_min.cpu()) / (self.visual_max.cpu() - self.visual_min.cpu() + 1e-8)
                    # Imputa usando DiCMoR (restituisce embedding)
                    textual_embd_imputed = self.model.impute_textual(visual_norm)
                    #textual_imputed = textual_embd_imputed.cpu().numpy()
                    textual_imputed_tensor = self.denormalize(textual_embd_imputed.unsqueeze(0), 'textual').squeeze(0)
                    textual_imputed = textual_imputed_tensor.cpu().numpy()
                output_path = os.path.join(output_textual_folder, item)
                np.save(output_path, textual_imputed)
            
            # Imputa missing visual da textual
            for item in tqdm(missing_visual, desc="Visual imputation"):
                if item in missing_textual:
                    # Usa la media globale invece di zeri
                    visual_imputed = self.visual_global_mean.cpu().numpy()
                    #visual_imputed = np.zeros(self.visual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica textual e imputa visual
                    textual_path = os.path.join(textual_folder, item + ".npy")
                    textual_feat = np.load(textual_path)
                    textual_tensor = torch.from_numpy(textual_feat).float()
                    # Normalizza
                    textual_norm = (textual_tensor - self.textual_min.cpu()) / (self.textual_max.cpu() - self.textual_min.cpu() + 1e-8)
                    # Imputa usando DiCMoR (restituisce embedding)
                    visual_embd_imputed = self.model.impute_visual(textual_norm)
                    #visual_imputed = visual_embd_imputed.cpu().numpy()
                    visual_imputed_tensor = self.denormalize(visual_embd_imputed.unsqueeze(0), 'visual').squeeze(0)
                    visual_imputed = visual_imputed_tensor.cpu().numpy()
                output_path = os.path.join(output_visual_folder, item)
                np.save(output_path, visual_imputed)
        print("IMPUTATION COMPLETED!")   

    def impute(self):
        
        self.set_seed()

        # Crea il modello DiCMoR
        self.model = DICMORModel(visual_dim=self.visual_dim, textual_dim=self.textual_dim, embd_size_v=self.embd_size_v, embd_size_t=self.embd_size_t, n_flow=self.n_flow, n_block=self.n_block, rcan_blocks=self.rcan_blocks, gpu_ids=self.gpu_ids)
        
        output_visual = self.output_paths[0] + '/dicmor'
        output_textual = self.output_paths[1] + '/dicmor'

        if self.missing_percentage_visual is not None:
            errors = {'visual':[], 'textual':[]}
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try: 
                missing_visual = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_dicmor")  #funzione in simple_imputation
                result_path_percentage_visual = os.path.join(output_visual, f"{percentage_visual}")
                os.makedirs(result_path_percentage_visual, exist_ok=True)   
                missing_visual_npy = []
                for item in missing_visual:
                    missing_visual_npy.append(f'{item}.npy')
                visual_folder = f"{self.dataset_path}/non_masked_visual_folder_{percentage_visual}"
                os.makedirs(visual_folder, exist_ok=True)
                for file in os.listdir(self.input_paths[0]):
                    if file.endswith('.npy') and (file not in missing_visual_npy): 
                       np.save(f'{visual_folder}/{file}', np.load(os.path.join(self.input_paths[0], file)))
            
                missing_textual = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_dicmor")  #funzione in simple_imputation
                result_path_percentage_textual = os.path.join(output_textual, f"{percentage_textual}")
                os.makedirs(result_path_percentage_textual, exist_ok=True)    
                missing_textual_npy = [] 
                for item in missing_textual:
                    missing_textual_npy.append(f'{item}.npy')
                textual_folder = f"{self.dataset_path}/non_masked_textual_folder_{percentage_textual}"
                os.makedirs(textual_folder, exist_ok=True)
                for file in os.listdir(self.input_paths[1]):
                    if file.endswith('.npy') and (file not in missing_textual_npy):
                       np.save(f'{textual_folder}/{file}', np.load(os.path.join(self.input_paths[1], file)))
                
                # 1. Carica dati
                visual_features, textual_features, all_present = self.load_data(visual_folder, textual_folder)
                # 2. Train DiCMoR
                self.train_dicmor(visual_features, textual_features, num_epochs=self.num_epochs , batch_size=self.batch_size, lr=self.learning_rate, log_p_weight=self.log_p_weight)
                # 3. Imputa missing modalities
                self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, result_path_percentage_visual, result_path_percentage_textual)

                # Calcola errore di ricostruzione
                mse_visual, cosine_sim_visual = self.compute_reconstruction_errors_for_modality(self.input_paths[0], result_path_percentage_visual, missing_visual_npy)
                errors['visual'].append((percentage_visual, mse_visual, cosine_sim_visual))
                mse_textual, cosine_sim_textual = self.compute_reconstruction_errors_for_modality(self.input_paths[1], result_path_percentage_textual, missing_textual_npy)
                errors['textual'].append((percentage_textual, mse_textual, cosine_sim_textual))

              finally:  
                    self.cleanup_temp_resources([visual_folder, textual_folder])

            return errors
        
        else:
            missing_visual_path = os.path.join(self.dataset_path, self.missing_visual_file)
            missing_visual = set(pd.read_csv(missing_visual_path, sep='\t', header=None)[0].to_list())  
            visual_folder = self.input_paths[0]

            missing_textual_path = os.path.join(self.dataset_path, self.missing_textual_file) 
            missing_textual = set(pd.read_csv(missing_textual_path, sep='\t', header=None)[0].to_list()) 
            textual_folder = self.input_paths[1]

            # 1. Carica dati
            visual_features, textual_features, all_present = self.load_data(visual_folder, textual_folder)
            # 2. Train DiCMoR
            self.train_dicmor(visual_features, textual_features, num_epochs=self.num_epochs , batch_size=self.batch_size, lr=self.learning_rate, log_p_weight=self.log_p_weight)
            # 3. Imputa missing modalities
            self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, output_visual, output_textual)