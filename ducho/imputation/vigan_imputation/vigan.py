#VIGAN Imputator per imputazione delle modalità mancanti
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ducho.imputation.vigan_imputation.viganmodel import VIGANModel
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

#classe figlia di ImputatorFather
class ViganImputator(ImputatorFather): #VIGAN-based imputation per multimodal recommendation, usa il codice estratto da https://github.com/chaoshangcs/VIGAN adattato per feature vectors (visual/textual embeddings)
    
    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, gpu_ids, seed, missing_percentage_visual, missing_percentage_textual, items_file, 
                 num_epochs_cyclegan, num_epochs_ae, num_epochs_joint, learning_rate, batch_size, beta1, lambda_A, lambda_B, lambda_idt, lambda_C, lambda_D, n_iter_G, n_iter_D): 
        """
        Args:
            dataset_path: path al dataset
            missing_visual_file: file con item ID mancanti visual
            missing_textual_file: file con item ID mancanti textual
            visual_dim: dimensione features visual (default: 2048 per ResNet50)
            textual_dim: dimensione features textual (default: 768 per MPNet)
            gpu_ids: lista GPU IDs ([0] per cuda:0, [] per CPU)
        """
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
    
        # Statistiche per normalizzazione
        self.visual_min = None
        self.visual_max = None
        self.textual_min = None
        self.textual_max = None
   
    def train_vigan(self, visual_features, textual_features, num_epochs_cyclegan, num_epochs_ae, num_epochs_joint, batch_size, lr, beta1, lambda_A, lambda_B, lambda_idt, lambda_C, lambda_D, n_iter_G, n_iter_D): #Training VIGAN in 3 fasi (come nel paper originale)   #60 100 100
        # Normalizza
        visual_norm, textual_norm = self.normalize_data(visual_features, textual_features)  #visual_features: numpy array [N, visual_dim] #textual_features: numpy array [N, textual_dim]
        # Setup optimizers
        self.model.setup_optimizers(lr=lr, beta1=beta1)   #beta1=0.5
        # Crea dataloader
        dataset = TensorDataset(visual_norm, textual_norm)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # FASE 1: Pre-train CycleGAN (G_A, G_B, D_A, D_B)      
        self.model.train()
        for epoch in range(num_epochs_cyclegan): #num_epochs_cyclegan: epochs per pre-train CycleGAN (FASE 1)
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
        
        # FASE 2: Pre-train Autoencoder
        for epoch in range(num_epochs_ae): #num_epochs_ae: epochs per pre-train Autoencoder (FASE 2)
            epoch_loss_AE = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters_pretrain_AE()
                epoch_loss_AE += self.model.loss_AE_pre.item()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_ae}] - " f"AE Loss: {epoch_loss_AE/len(dataloader):.4f}")

        # FASE 3: Joint Optimization (AE + G_A + G_B)       
        for epoch in range(num_epochs_joint): #num_epochs_joint: epochs per joint optimization (FASE 3)
            epoch_loss = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters(lambda_C=lambda_C, lambda_D=lambda_D, n_iter_G=n_iter_G, n_iter_D=n_iter_D)
                epoch_loss += self.model.loss_AE_GA_GB.item()
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_joint}] - " f"Joint Loss: {epoch_loss/len(dataloader):.4f}")
  
    def impute_missing(self, visual_folder, textual_folder, missing_visual, missing_textual, output_visual_folder, output_textual_folder): #Imputa le modalità mancanti usando il modello trainato
        os.makedirs(output_visual_folder, exist_ok=True) #output_visual_folder: folder output per visual imputate
        os.makedirs(output_textual_folder, exist_ok=True) #output_textual_folder: folder output per textual imputate
        self.model.eval()

        with torch.no_grad():
            # Imputa missing textual da visual
            for item in tqdm(missing_textual, desc="Textual imputation"): #missing_textual: set di item IDs con textual mancante
                if item in missing_visual:
                    # Usa la media globale invece di zeri
                    textual_imputed = self.textual_global_mean.cpu().numpy()
                    #textual_imputed = np.zeros(self.textual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica visual e imputa textual
                    visual_path = os.path.join(visual_folder, item + ".npy")  #visual_folder: folder con visual features originali
                    visual_feat = np.load(visual_path)
                    visual_tensor = torch.from_numpy(visual_feat).float().unsqueeze(0)
                    # Normalizza
                    visual_norm = (visual_tensor - self.visual_min.cpu()) / (self.visual_max.cpu() - self.visual_min.cpu() + 1e-8)
                    if visual_norm.ndim == 3:
                       visual_norm = visual_norm.squeeze(1)
                    elif visual_norm.ndim == 1:
                        visual_norm = visual_norm.unsqueeze(0)
                    visual_norm = visual_norm.to(self.device)
                    # Imputa usando Generator G_A (visual -> textual)
                    textual_fake_norm = self.model.netG_A(visual_norm)
                    # Raffina con Autoencoder
                    _, textual_refined_norm = self.model.AE(visual_norm, textual_fake_norm)
                    # Media pesata
                    textual_imputed_norm = 0.5 * textual_fake_norm + 0.5 * textual_refined_norm
                    # Denormalizza
                    textual_imputed_tensor = self.denormalize(textual_imputed_norm, 'textual')
                    textual_imputed = textual_imputed_tensor.squeeze().cpu().numpy()
                # Salva
                output_path = os.path.join(output_textual_folder, item)
                np.save(output_path, textual_imputed)
            
            # Imputa missing visual da textual
            for item in tqdm(missing_visual, desc="Visual imputation"):
                if item in missing_textual:  #missing_textual: set di item IDs con textual mancante
                    # Usa la media globale invece di zeri
                    visual_imputed = self.visual_global_mean.cpu().numpy()
                    #visual_imputed = np.zeros(self.visual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica textual e imputa visual
                    textual_path = os.path.join(textual_folder, item + ".npy") #textual_folder: folder con textual features originali
                    textual_feat = np.load(textual_path)
                    textual_tensor = torch.from_numpy(textual_feat).float().unsqueeze(0)
                    # Normalizza
                    textual_norm = (textual_tensor - self.textual_min.cpu()) / (self.textual_max.cpu() - self.textual_min.cpu() + 1e-8)
                    if textual_norm.ndim == 3:
                       textual_norm = textual_norm.squeeze(1)
                    elif textual_norm.ndim == 1:
                        textual_norm = textual_norm.unsqueeze(0)
                    textual_norm = textual_norm.to(self.device)
                    # Imputa usando Generator G_B (textual -> visual)
                    visual_fake_norm = self.model.netG_B(textual_norm)
                    # Raffina con Autoencoder
                    visual_refined_norm, _ = self.model.AE(visual_fake_norm, textual_norm)
                    # Media pesata
                    visual_imputed_norm = 0.5 * visual_fake_norm + 0.5 * visual_refined_norm
                    # Denormalizza
                    visual_imputed_tensor = self.denormalize(visual_imputed_norm, 'visual')
                    visual_imputed = visual_imputed_tensor.squeeze().cpu().numpy()
                # Salva
                output_path = os.path.join(output_visual_folder, item)
                np.save(output_path, visual_imputed)
            
    def impute(self): #Pipeline completa: carica dati, train, imputa

        self.set_seed()

        # Crea il modello VIGAN
        self.model = VIGANModel(input_A_dim=self.visual_dim, input_B_dim=self.textual_dim, gpu_ids=self.gpu_ids, use_lsgan=True)
        
        output_visual = self.output_paths[0] + '/vigan'
        output_textual = self.output_paths[1] + '/vigan'

        if self.missing_percentage_visual is not None:
            errors = {'visual':[], 'textual':[]}
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try:
                missing_visual = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_vigan")  #funzione in simple_imputation
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
            
                missing_textual = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_vigan")  #funzione in simple_imputation
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
                # 2. Train VIGAN
                self.train_vigan(visual_features, textual_features, num_epochs_cyclegan=self.num_epochs_cyclegan, num_epochs_ae=self.num_epochs_ae, num_epochs_joint=self.num_epochs_joint, batch_size=self.batch_size, lr=self.learning_rate,
                                beta1=self.beta1, lambda_A=self.lambda_A, lambda_B=self.lambda_B, lambda_idt=self.lambda_idt, lambda_C=self.lambda_C, lambda_D=self.lambda_D, n_iter_G=self.n_iter_G, n_iter_D=self.n_iter_D)
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
            # 2. Train VIGAN
            self.train_vigan(visual_features, textual_features, num_epochs_cyclegan=self.num_epochs_cyclegan, num_epochs_ae=self.num_epochs_ae, num_epochs_joint=self.num_epochs_joint, batch_size=self.batch_size, lr=self.learning_rate, 
                            beta1=self.beta1, lambda_A=self.lambda_A, lambda_B=self.lambda_B, lambda_idt=self.lambda_idt, lambda_C=self.lambda_C, lambda_D=self.lambda_D, n_iter_G=self.n_iter_G, n_iter_D=self.n_iter_D)
            # 3. Imputa missing modalities
            self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, output_visual, output_textual)
    








"""
#VIGAN Imputator per imputazione delle modalità mancanti
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ducho.imputation.vigan_imputation.viganmodel import VIGANModel
import random 
from ducho.imputation.simple_imputation import compute_artificial_missing_items, compute_reconstruction_errors


class ViganImputator: #VIGAN-based imputation per multimodal recommendation, usa il codice estratto da https://github.com/chaoshangcs/VIGAN adattato per feature vectors (visual/textual embeddings)
    
    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, visual_dim, textual_dim, gpu_ids, seed, missing_percentage_visual, missing_percentage_textual, items_file):  
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
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
    
        # Statistiche per normalizzazione
        self.visual_min = None
        self.visual_max = None
        self.textual_min = None
        self.textual_max = None
    
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
        return visual_norm, textual_norm
    
    def denormalize(self, data, modality='visual'):  #Denormalizza da [0,1] al range originale
        if modality == 'visual':
            return data * (self.visual_max - self.visual_min) + self.visual_min
        else:
            return data * (self.textual_max - self.textual_min) + self.textual_min
    
    def train_vigan(self, visual_features, textual_features, num_epochs_cyclegan=60, num_epochs_ae=100, num_epochs_joint=100, batch_size=2, lr=0.0002): #Training VIGAN in 3 fasi (come nel paper originale)
        # Normalizza
        visual_norm, textual_norm = self.normalize_data(visual_features, textual_features)  #visual_features: numpy array [N, visual_dim] #textual_features: numpy array [N, textual_dim
        # Setup optimizers
        self.model.setup_optimizers(lr=lr, beta1=0.5)
        # Crea dataloader
        dataset = TensorDataset(visual_norm, textual_norm)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # FASE 1: Pre-train CycleGAN (G_A, G_B, D_A, D_B)      
        self.model.train()
        for epoch in range(num_epochs_cyclegan): #num_epochs_cyclegan: epochs per pre-train CycleGAN (FASE 1)
            epoch_loss_G = 0
            epoch_loss_D_A = 0
            epoch_loss_D_B = 0
            
            for real_A, real_B in dataloader: 
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters_pretrain_cycleGAN(lambda_A=10.0, lambda_B=10.0, lambda_idt=0.5)
                epoch_loss_G += self.model.loss_G.item()
                epoch_loss_D_A += self.model.loss_D_A.item()
                epoch_loss_D_B += self.model.loss_D_B.item()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_cyclegan}] - " f"G: {epoch_loss_G/len(dataloader):.4f}, " f"D_A: {epoch_loss_D_A/len(dataloader):.4f}, " f"D_B: {epoch_loss_D_B/len(dataloader):.4f}")
        
        # FASE 2: Pre-train Autoencoder
        for epoch in range(num_epochs_ae): #num_epochs_ae: epochs per pre-train Autoencoder (FASE 2)
            epoch_loss_AE = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters_pretrain_AE()
                epoch_loss_AE += self.model.loss_AE_pre.item()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_ae}] - " f"AE Loss: {epoch_loss_AE/len(dataloader):.4f}")

        # FASE 3: Joint Optimization (AE + G_A + G_B)       
        for epoch in range(num_epochs_joint): #num_epochs_joint: epochs per joint optimization (FASE 3)
            epoch_loss = 0
            
            for real_A, real_B in dataloader:
                self.model.set_input(real_A, real_B)
                self.model.optimize_parameters(lambda_C=1.0, lambda_D=1.0, n_iter_G=2, n_iter_D=1)
                epoch_loss += self.model.loss_AE_GA_GB.item()
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs_joint}] - " f"Joint Loss: {epoch_loss/len(dataloader):.4f}")
  
    def impute_missing(self, visual_folder, textual_folder, missing_visual, missing_textual, output_visual_folder, output_textual_folder): #Imputa le modalità mancanti usando il modello trainato
        os.makedirs(output_visual_folder, exist_ok=True) #output_visual_folder: folder output per visual imputate
        os.makedirs(output_textual_folder, exist_ok=True) #output_textual_folder: folder output per textual imputate
        self.model.eval()
        with torch.no_grad():
            
            print(f"\nImputing missing textual features...") # Imputa missing textual da visual
            for item in tqdm(missing_textual, desc="Textual imputation"): #missing_textual: set di item IDs con textual mancante
                if item in missing_visual: # missing_visual: set di item IDs con visual mancante 
                    textual_imputed = np.zeros(self.textual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica visual e imputa textual
                    visual_path = os.path.join(visual_folder, item + ".npy")  #visual_folder: folder con visual features originali
                    visual_feat = np.load(visual_path)
                    visual_tensor = torch.from_numpy(visual_feat).float().unsqueeze(0)
                    # Normalizza
                    visual_norm = (visual_tensor - self.visual_min.cpu()) / (self.visual_max.cpu() - self.visual_min.cpu() + 1e-8)
                    if visual_norm.ndim == 3:
                       visual_norm = visual_norm.squeeze(1)
                    elif visual_norm.ndim == 1:
                        visual_norm = visual_norm.unsqueeze(0)
                    visual_norm = visual_norm.to(self.device)
                    # Imputa usando Generator G_A (visual -> textual)
                    textual_fake_norm = self.model.netG_A(visual_norm)
                    # Raffina con Autoencoder
                    _, textual_refined_norm = self.model.AE(visual_norm, textual_fake_norm)
                    # Media pesata
                    textual_imputed_norm = 0.5 * textual_fake_norm + 0.5 * textual_refined_norm
                    # Denormalizza
                    textual_imputed_tensor = self.denormalize(textual_imputed_norm, 'textual')
                    textual_imputed = textual_imputed_tensor.squeeze().cpu().numpy()
                # Salva
                output_path = os.path.join(output_textual_folder, item)
                np.save(output_path, textual_imputed)
            
            print(f"\nImputing missing visual features...") # Imputa missing visual da textual
            for item in tqdm(missing_visual, desc="Visual imputation"):
                if item in missing_textual:  #missing_textual: set di item IDs con textual mancante
                    visual_imputed = np.zeros(self.visual_dim, dtype=np.float32) # Entrambe mancanti -> usa zeros
                else:
                    # Carica textual e imputa visual
                    textual_path = os.path.join(textual_folder, item + ".npy") #textual_folder: folder con textual features originali
                    textual_feat = np.load(textual_path)
                    textual_tensor = torch.from_numpy(textual_feat).float().unsqueeze(0)
                    # Normalizza
                    textual_norm = (textual_tensor - self.textual_min.cpu()) / (self.textual_max.cpu() - self.textual_min.cpu() + 1e-8)
                    if textual_norm.ndim == 3:
                       textual_norm = textual_norm.squeeze(1)
                    elif textual_norm.ndim == 1:
                        textual_norm = textual_norm.unsqueeze(0)
                    textual_norm = textual_norm.to(self.device)
                    # Imputa usando Generator G_B (textual -> visual)
                    visual_fake_norm = self.model.netG_B(textual_norm)
                    # Raffina con Autoencoder
                    visual_refined_norm, _ = self.model.AE(visual_fake_norm, textual_norm)
                    # Media pesata
                    visual_imputed_norm = 0.5 * visual_fake_norm + 0.5 * visual_refined_norm
                    # Denormalizza
                    visual_imputed_tensor = self.denormalize(visual_imputed_norm, 'visual')
                    visual_imputed = visual_imputed_tensor.squeeze().cpu().numpy()
                # Salva
                output_path = os.path.join(output_visual_folder, item)
                np.save(output_path, visual_imputed)
        print("IMPUTATION COMPLETED!")

        if self.missing_percentage_visual is not None: #per l'errore di ricostruzione
          missing_visual_npy = []
          for item in missing_visual:
             missing_visual_npy.append(f'{item}.npy')
          real_embeddings_visual = []
          for file in os.listdir(self.input_paths[0]):
             if file.endswith('.npy') and (file in missing_visual_npy):
                 real_embeddings_visual.append(np.load(os.path.join(self.input_paths[0], file)))

          imputed_embeddings_visual = []
          for file in os.listdir(output_visual_folder):
             if file.endswith('.npy') and (file in missing_visual_npy):
                 imputed_embeddings_visual.append(np.load(os.path.join(output_visual_folder, file)))
         
          mse_error_visual = compute_reconstruction_errors(real_embeddings_visual, imputed_embeddings_visual)

          missing_textual_npy = []
          for item in missing_textual:
             missing_textual_npy.append(f'{item}.npy')
          real_embeddings_textual = []
          for file in os.listdir(self.input_paths[1]):
             if file.endswith('.npy') and (file in missing_textual_npy):
                 real_embeddings_textual.append(np.load(os.path.join(self.input_paths[1], file)))

          imputed_embeddings_textual = []
          for file in os.listdir(output_textual_folder):
             if file.endswith('.npy') and (file in missing_textual_npy):
                 imputed_embeddings_textual.append(np.load(os.path.join(output_textual_folder, file)))
         
          mse_error_textual = compute_reconstruction_errors(real_embeddings_textual, imputed_embeddings_textual)

          return [mse_error_visual, mse_error_textual]
            
     
    def impute(self): #Pipeline completa: carica dati, train, imputa

        #settare seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Crea il modello VIGAN
        self.model = VIGANModel(input_A_dim=self.visual_dim, input_B_dim=self.textual_dim, gpu_ids=self.gpu_ids, use_lsgan=True)
        
        output_visual = self.output_paths[0] + '/vigan'
        output_textual = self.output_paths[1] + '/vigan'

        if self.missing_percentage_visual is not None:
            missing_visual = compute_artificial_missing_items(self.dataset_path, self.missing_percentage_visual, self.items_file, self.seed, file_name='artificial_missing_visual_file')  #funzione in simple_imputation
            missing_visual_npy = []
            for item in missing_visual:
               missing_visual_npy.append(f'{item}.npy')
            visual_folder = 'local/data/demo_giulia/non_masked_visual_folder'
            os.makedirs(visual_folder, exist_ok=True)
            for file in os.listdir(self.input_paths[0]):
                if file.endswith('.npy') and (file not in missing_visual_npy): 
                   np.save(f'{visual_folder}/{file}', np.load(os.path.join(self.input_paths[0], file)))
            
            missing_textual = compute_artificial_missing_items(self.dataset_path, self.missing_percentage_textual, self.items_file, self.seed, file_name='artificial_missing_textual_file')  #funzione in simple_imputation
            missing_textual_npy = [] 
            for item in missing_textual:
               missing_textual_npy.append(f'{item}.npy')
            textual_folder = 'local/data/demo_giulia/non_masked_textual_folder'
            os.makedirs(textual_folder, exist_ok=True)
            for file in os.listdir(self.input_paths[1]):
                if file.endswith('.npy') and (file not in missing_textual_npy):
                   np.save(f'{textual_folder}/{file}', np.load(os.path.join(self.input_paths[1], file)))
        
        else: 
            missing_visual_path = os.path.join(self.dataset_path, self.missing_visual_file)
            missing_visual = set(pd.read_csv(missing_visual_path, sep='\t', header=None)[0].to_list())  
            visual_folder = self.input_paths[0]

            missing_textual_path = os.path.join(self.dataset_path, self.missing_textual_file) 
            missing_textual = set(pd.read_csv(missing_textual_path, sep='\t', header=None)[0].to_list()) 
            textual_folder = self.input_paths[1]

        # 1. Carica dati
        visual_features, textual_features, all_present = self.load_data(visual_folder, textual_folder)
        # 2. Train VIGAN
        self.train_vigan(visual_features, textual_features, num_epochs_cyclegan=100, num_epochs_ae=100, num_epochs_joint=300, batch_size=2, lr=0.0002)  #batchsize 64
        # 3. Imputa missing modalities
        list_error_v_t = self.impute_missing(visual_folder, textual_folder, missing_visual, missing_textual, output_visual, output_textual)

        return list_error_v_t
    
"""