import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ducho.imputation.vigan_imputation.viganmodel import VIGANModel
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

# Child class of ImputatorFather
class ViganImputator(ImputatorFather):
    
    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, 
                 visual_dim, textual_dim, gpu_ids, seed, missing_percentage_visual, missing_percentage_textual, 
                 items_file, num_epochs_cyclegan, num_epochs_ae, num_epochs_joint, learning_rate, batch_size, 
                 beta1, lambda_A, lambda_B, lambda_idt, lambda_C, lambda_D, n_iter_G, n_iter_D): 
        
        super().__init__()
        self.dataset_path = dataset_path
        self.missing_visual_file = missing_visual_file
        self.missing_textual_file = missing_textual_file
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.gpu_ids = gpu_ids
        self.seed = seed  
        self.missing_percentage_visual = missing_percentage_visual
        self.missing_percentage_textual = missing_percentage_textual
        self.items_file = items_file
        self.num_epochs_cyclegan = num_epochs_cyclegan
        self.num_epochs_ae = num_epochs_ae
        self.num_epochs_joint = num_epochs_joint
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta1 = beta1
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt = lambda_idt
        self.lambda_C = lambda_C
        self.lambda_D = lambda_D
        self.n_iter_G = n_iter_G
        self.n_iter_D = n_iter_D
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
    
        # Statistics for normalization
        self.visual_min = None
        self.visual_max = None
        self.textual_min = None
        self.textual_max = None

   # Three-phase Vigan training (as in the original paper)
    def train_vigan(self, visual_features, textual_features, num_epochs_cyclegan, num_epochs_ae, num_epochs_joint, 
                    batch_size, lr, beta1, lambda_A, lambda_B, lambda_idt, lambda_C, lambda_D, n_iter_G, n_iter_D): 
        # Normalize
        visual_norm, textual_norm = self.normalize_data(visual_features, textual_features)  # visual_features: numpy array [N, visual_dim] - textual_features: numpy array [N, textual_dim]
        # Setup optimizers
        self.model.setup_optimizers(lr=lr, beta1=beta1)   # Parameter: beta1=0.5
        # Create dataloader
        dataset = TensorDataset(visual_norm, textual_norm)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # PHASE 1: Pre-train CycleGAN (G_A, G_B, D_A, D_B)      
        self.model.train()
        for epoch in range(num_epochs_cyclegan):   # num_epochs_cyclegan: epochs for pre-training the CycleGAN
            epoch_loss_G = 0
            epoch_loss_D_A = 0
            epoch_loss_D_B = 0
            
            for real_A, real_B in dataloader: 
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters_pretrain_cycleGAN(lambda_A=lambda_A, lambda_B=lambda_B, lambda_idt=lambda_idt)
                epoch_loss_G += self.model.loss_G.item()
                epoch_loss_D_A += self.model.loss_D_A.item()
                epoch_loss_D_B += self.model.loss_D_B.item()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_cyclegan}] - " f"G: {epoch_loss_G/len(dataloader):.4f}, " f"D_A: {epoch_loss_D_A/len(dataloader):.4f}, " f"D_B: {epoch_loss_D_B/len(dataloader):.4f}")
        
        # PHASE 2: Pre-train Autoencoder
        for epoch in range(num_epochs_ae):          # num_epochs_ae: epochs for pre-training the Autoencoder 
            epoch_loss_AE = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters_pretrain_AE()
                epoch_loss_AE += self.model.loss_AE_pre.item()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_ae}] - " f"AE Loss: {epoch_loss_AE/len(dataloader):.4f}")

        # PHASE 3: Joint Optimization (AE + G_A + G_B)       
        for epoch in range(num_epochs_joint):        # num_epochs_joint: epochs for joint optimization
            epoch_loss = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters(lambda_C=lambda_C, lambda_D=lambda_D, n_iter_G=n_iter_G, n_iter_D=n_iter_D)
                epoch_loss += self.model.loss_AE_GA_GB.item()
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_joint}] - " f"Joint Loss: {epoch_loss/len(dataloader):.4f}")

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
                    #textual_imputed = np.zeros(self.textual_dim, dtype=np.float32) # Both missing -> use zeros
                else:
                    # Load visual and impute textual
                    visual_path = os.path.join(visual_folder, item + ".npy")
                    visual_feat = np.load(visual_path)
                    visual_tensor = torch.from_numpy(visual_feat).float().unsqueeze(0)
                    # Normalize
                    visual_norm = (visual_tensor - self.visual_min.cpu()) / (self.visual_max.cpu() - self.visual_min.cpu() + 1e-8)
                    if visual_norm.ndim == 3:
                       visual_norm = visual_norm.squeeze(1)
                    elif visual_norm.ndim == 1:
                        visual_norm = visual_norm.unsqueeze(0)
                    visual_norm = visual_norm.to(self.device)
                    # Impute using Generator G_A (visual -> textual)
                    textual_fake_norm = self.model.netG_A(visual_norm)
                    # Refine with Autoencoder
                    _, textual_refined_norm = self.model.AE(visual_norm, textual_fake_norm)
                    # Weighted average
                    textual_imputed_norm = 0.5 * textual_fake_norm + 0.5 * textual_refined_norm
                    # Denormalize
                    textual_imputed_tensor = self.denormalize(textual_imputed_norm, 'textual')
                    textual_imputed = textual_imputed_tensor.squeeze().cpu().numpy()
                # Save
                output_path = os.path.join(output_textual_folder, item)
                np.save(output_path, textual_imputed)
            
            # Impute missing visual from textual     
            for item in tqdm(missing_visual, desc="Visual imputation"):
                if item in missing_textual:  
                    # Use the global average instead of zeros
                    visual_imputed = self.visual_global_mean.cpu().numpy()
                    #visual_imputed = np.zeros(self.visual_dim, dtype=np.float32) # Both missing -> use zeros
                else:
                    # Load textual and impute visual
                    textual_path = os.path.join(textual_folder, item + ".npy")
                    textual_feat = np.load(textual_path)
                    textual_tensor = torch.from_numpy(textual_feat).float().unsqueeze(0)
                    # Normalize
                    textual_norm = (textual_tensor - self.textual_min.cpu()) / (self.textual_max.cpu() - self.textual_min.cpu() + 1e-8)
                    if textual_norm.ndim == 3:
                       textual_norm = textual_norm.squeeze(1)
                    elif textual_norm.ndim == 1:
                        textual_norm = textual_norm.unsqueeze(0)
                    textual_norm = textual_norm.to(self.device)
                    # Impute using Generator G_B (textual -> visual)
                    visual_fake_norm = self.model.netG_B(textual_norm)
                    # Refine with Autoencoder
                    visual_refined_norm, _ = self.model.AE(visual_fake_norm, textual_norm)
                    # Weighted average
                    visual_imputed_norm = 0.5 * visual_fake_norm + 0.5 * visual_refined_norm
                    # Denormalize
                    visual_imputed_tensor = self.denormalize(visual_imputed_norm, 'visual')
                    visual_imputed = visual_imputed_tensor.squeeze().cpu().numpy()
                # Save
                output_path = os.path.join(output_visual_folder, item)
                np.save(output_path, visual_imputed)
            
    def impute(self):

        self.set_seed()

        # Create the VIGAN model
        self.model = VIGANModel(input_A_dim=self.visual_dim, input_B_dim=self.textual_dim, gpu_ids=self.gpu_ids, use_lsgan=True)
        
        output_visual = self.output_paths[0] + '/vigan'
        output_textual = self.output_paths[1] + '/vigan'

        if self.missing_percentage_visual is not None:
            errors = {'visual':[], 'textual':[]}
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try:
                missing_visual = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_vigan")  
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
            
                missing_textual = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_vigan")  
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
                # Train VIGAN
                self.train_vigan(visual_features, textual_features, num_epochs_cyclegan=self.num_epochs_cyclegan, num_epochs_ae=self.num_epochs_ae, num_epochs_joint=self.num_epochs_joint, batch_size=self.batch_size, lr=self.learning_rate,
                                beta1=self.beta1, lambda_A=self.lambda_A, lambda_B=self.lambda_B, lambda_idt=self.lambda_idt, lambda_C=self.lambda_C, lambda_D=self.lambda_D, n_iter_G=self.n_iter_G, n_iter_D=self.n_iter_D)
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
            # Train VIGAN
            self.train_vigan(visual_features, textual_features, num_epochs_cyclegan=self.num_epochs_cyclegan, num_epochs_ae=self.num_epochs_ae, num_epochs_joint=self.num_epochs_joint, batch_size=self.batch_size, lr=self.learning_rate, 
                            beta1=self.beta1, lambda_A=self.lambda_A, lambda_B=self.lambda_B, lambda_idt=self.lambda_idt, lambda_C=self.lambda_C, lambda_D=self.lambda_D, n_iter_G=self.n_iter_G, n_iter_D=self.n_iter_D)
            # Impute missing modalities
            self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, output_visual, output_textual)