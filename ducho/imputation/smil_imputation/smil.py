import torch
import os 
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ducho.imputation.smil_imputation.smilmodel import SMILModel
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# Child class of ImputatorFather
class SmilImputator(ImputatorFather):
    
    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths,
                 visual_dim, textual_dim, gpu_ids, seed, missing_percentage_visual, missing_percentage_textual, 
                 items_file, num_epochs, learning_rate, batch_size, alpha, beta):
        
        super().__init__()
        self.dataset_path = dataset_path 
        self.missing_visual_file = missing_visual_file 
        self.missing_textual_file = missing_textual_file 
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.seed = seed
        self.missing_percentage_visual = missing_percentage_visual
        self.missing_percentage_textual = missing_percentage_textual
        self.items_file = items_file

        if isinstance(visual_dim, (tuple, list)):  
            visual_dim = visual_dim[0]
        if isinstance(textual_dim, (tuple, list)):  
            textual_dim = textual_dim[0]

        self.visual_dim = int(visual_dim)   
        self.textual_dim = int(textual_dim)  
        self.gpu_ids = gpu_ids  
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')     
        
        # Statistics for normalization
        self.visual_min = None
        self.visual_max = None
        self.textual_min = None
        self.textual_max = None

    # Training SMIL with meta-learning
    def train_smil(self, visual_features, textual_features, num_epochs, batch_size, lr):
        # Normalize
        visual_norm, textual_norm = self.normalize_data(visual_features, textual_features)
        # Compute averages of modalities
        self.model.compute_modality_means(visual_norm, textual_norm)   
        # Setup optimizers
        self.model.setup_optimizers(lr=lr) 
        # Create dataloader
        dataset = TensorDataset(visual_norm, textual_norm)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
       
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):  # Parameters: num_epochs=60, batch_size=32, lr=0.001)
            epoch_loss_textual = 0
            epoch_loss_visual = 0
            num_batches = 0
            for visual_batch, textual_batch in dataloader:
                loss_textual = self.model.train_step(visual_batch, textual_batch, missing_modality='textual')   # Train step 1: simulate missing textual 
                loss_visual = self.model.train_step(visual_batch, textual_batch, missing_modality='visual')     # Train step 2: simulate missing visual
                epoch_loss_textual += loss_textual
                epoch_loss_visual += loss_visual
                num_batches += 1
            avg_loss_textual = epoch_loss_textual / num_batches
            avg_loss_visual = epoch_loss_visual / num_batches
            avg_loss_total = (avg_loss_textual + avg_loss_visual) / 2
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - " f"Loss (textual): {avg_loss_textual:.4f}, " f"Loss (visual): {avg_loss_visual:.4f}, " f"Loss (total): {avg_loss_total:.4f}")

    # Impute missing modalities using the trained model        
    def impute_missing(self, visual_folder, textual_folder, missing_visual, missing_textual, output_visual_folder, output_textual_folder): 

        os.makedirs(output_visual_folder, exist_ok=True)
        os.makedirs(output_textual_folder, exist_ok=True)
        self.model.eval()

        with torch.no_grad():
            # Impute missing textual from visual
            for item in tqdm(missing_textual, desc="Textual imputation"):
                if item in missing_visual:
                    # Use the global average instead of zeros
                    textual_imputed = self.textual_global_mean.cpu().numpy()
                    #textual_imputed = np.zeros(self.textual_dim, dtype=np.float32)  # Both missing -> use zeros
                else:
                    # Load visual and impute textual
                    visual_path = os.path.join(visual_folder, item + '.npy')
                    visual_feat = np.load(visual_path)
                    visual_tensor = torch.from_numpy(visual_feat).float() 
                    # Normalize
                    visual_norm = (visual_tensor - self.visual_min.cpu()) / (self.visual_max.cpu() - self.visual_min.cpu() + 1e-8)    
                    textual_imputed_norm = self.model.impute_textual(visual_norm)
                    textual_imputed_norm = torch.clamp(textual_imputed_norm, 0.0, 1.0)
                    textual_imputed_tensor = self.denormalize(textual_imputed_norm.unsqueeze(0), 'textual').squeeze(0)
                    textual_imputed = textual_imputed_tensor.cpu().numpy()

                output_path = os.path.join(output_textual_folder, item)
                np.save(output_path, textual_imputed)
            
            # Impute missing visual from textual     
            for item in tqdm(missing_visual, desc="Visual imputation"):
                if item in missing_textual:
                    # Use the global average instead of zeros
                    visual_imputed = self.visual_global_mean.cpu().numpy()
                    #visual_imputed = np.zeros(self.visual_dim, dtype=np.float32)  # Both missing -> use zeros
                else:
                    # Load textual and impute visual
                    textual_path = os.path.join(textual_folder, item + '.npy')
                    textual_feat = np.load(textual_path)
                    textual_tensor = torch.from_numpy(textual_feat).float()
                    # Normalize
                    textual_norm = (textual_tensor - self.textual_min.cpu()) / (self.textual_max.cpu() - self.textual_min.cpu() + 1e-8)
                    visual_imputed_norm = self.model.impute_visual(textual_norm)
                    visual_imputed_norm = torch.clamp(visual_imputed_norm, 0.0, 1.0)
                    visual_imputed_tensor = self.denormalize(visual_imputed_norm.unsqueeze(0), 'visual').squeeze(0)
                    visual_imputed = visual_imputed_tensor.cpu().numpy()
                    
                output_path = os.path.join(output_visual_folder, item)
                np.save(output_path, visual_imputed)

        print("IMPUTATION COMPLETED!")

        
    def impute(self):

        self.set_seed()

        # Create SMIL model
        self.model = SMILModel(visual_dim=self.visual_dim, textual_dim=self.textual_dim, gpu_ids=self.gpu_ids, alpha=self.alpha, beta=self.beta)  

        output_visual = self.output_paths[0] + '/smil'
        output_textual = self.output_paths[1] + '/smil'

        if self.missing_percentage_visual is not None:
            errors = {'visual':[], 'textual':[]}
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try:
                missing_visual = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_smil")  
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
            
                missing_textual = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_smil")  
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
                
                # Load data
                visual_features, textual_features, all_present = self.load_data(visual_folder, textual_folder)
                # Train SMIL
                self.train_smil(visual_features, textual_features, num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.learning_rate)
                # Impute missing modalities
                self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, result_path_percentage_visual, result_path_percentage_textual)

                # Compute recontruction error
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
        
            # Load data
            visual_features, textual_features, all_present = self.load_data(visual_folder, textual_folder)
            # Train SMIL
            self.train_smil(visual_features, textual_features, num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.learning_rate)
            # Impute missing modalities
            self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, output_visual, output_textual)