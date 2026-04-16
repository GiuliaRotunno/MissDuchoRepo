import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ducho.internal.father_classes.ImputatorFather import ImputatorFather

class VAEVisual2Textual(nn.Module):
    def __init__(self):
        super(VAEVisual2Textual, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
  
        self.fc_mu = nn.Linear(512, 256)
        self.fc_logvar = nn.Linear(512, 256)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class VAETextual2Visual(nn.Module):
    def __init__(self):
        super(VAETextual2Visual, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, 256)
        self.fc_logvar = nn.Linear(512, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
# VAE -> trains a variational autoencoder to reconstruct missing features from one modality m1 through the available features from another modality m2
#classe figlia di ImputatorFather
class VarautoencImputator(ImputatorFather):    

    def __init__(self, dataset_path, missing_visual_file, missing_textual_file, input_paths, output_paths, seed, missing_percentage_visual, missing_percentage_textual, items_file, num_epochs, learning_rate, batch_size, beta_visual, beta_textual):
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
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta_visual = beta_visual
        self.beta_textual = beta_textual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    def _train_varautoencoder_textual(self, visual_normalized, textual_normalized):  #Addestra vae per ricostruire textual da visual
      
            model = VAEVisual2Textual()
            model.to(self.device)
            criterion = nn.BCELoss(reduction='mean')  # necessario per KL
            optimizer = optim.Adagrad(model.parameters(), lr=self.learning_rate)  #lr=0.01 default
            dataset = TensorDataset(visual_normalized, textual_normalized)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) #batch_size=32 deafult
            beta = self.beta_textual     #beta = 1.0 

            cumulative_loss = 0
            for epoch in range(self.num_epochs):  #num_epochs=1000 default
                for inputs, targets in dataloader:
                        outputs, mu, logvar = model(inputs)  
                        recon_loss = criterion(outputs, targets)
                        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + beta * kl_loss 
                        cumulative_loss += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if (epoch + 1) % (self.num_epochs // 10) == 0:
                    avg_loss_per_epoch = cumulative_loss / (self.num_epochs // 10)  # loss media per epoch 
                    print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss_per_epoch:.4f}")
                    cumulative_loss = 0
            
            return model
    
    def _train_varautoencoder_visual(self, textual_normalized, visual_normalized):  #Addestra vae per ricostruire visual da textual

            model = VAETextual2Visual()
            model.to(self.device)
            criterion = nn.BCELoss(reduction='mean')
            optimizer = optim.Adagrad(model.parameters(), lr=self.learning_rate)
            dataset = TensorDataset(textual_normalized, visual_normalized)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            beta = self.beta_visual    #beta = 0.01 

            cumulative_loss = 0
            for epoch in range(self.num_epochs):
                for inputs, targets in dataloader:
                    outputs, mu, logvar = model(inputs)  
                    recon_loss = criterion(outputs, targets)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss  
                    cumulative_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if (epoch + 1) % (self.num_epochs // 10) == 0:
                    avg_loss_per_epoch = cumulative_loss / (self.num_epochs // 10)  # loss media per epoch 
                    print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss_per_epoch:.4f}")
                    cumulative_loss = 0
            
            return model

    def _impute_textual_modality(self, missing_textual, missing_visual, visual_folder, model_impute_textual, visual_min, visual_max, textual_min, textual_max, textual_global_mean, output_path):   #Imputa la modalità textual mancante
     
            model_impute_textual.eval()
            with torch.no_grad():
                # impute missing textual from visual
                for miss in missing_textual:
                    if miss in missing_visual:
                        # Usa la media globale denormalizzata invece di zeri
                        output = (textual_global_mean * (textual_max - textual_min)) + textual_min
                        np.save(os.path.join(output_path, f'{miss}.npy'), output.detach().cpu().numpy())
                    else:
                        visual_input = torch.from_numpy(np.load(os.path.join(visual_folder, f'{miss}.npy')))
                        visual_input = (visual_input - visual_min.cpu()) / (visual_max.cpu() - visual_min.cpu() + 1e-8)
                        visual_input = visual_input.to(self.device)
                        output, _, _ = model_impute_textual(visual_input)  
                        output = (output * (textual_max - textual_min)) + textual_min
                        np.save(os.path.join(output_path, f'{miss}.npy'),output.detach().cpu().numpy())

    def _impute_visual_modality(self, missing_visual, missing_textual, textual_folder, model_impute_visual, textual_min, textual_max, visual_min, visual_max, visual_global_mean, output_path):  #Imputa la modalità visual mancante
        
        model_impute_visual.eval()
        with torch.no_grad():
                #impute missing visual from textual 
                for miss in missing_visual:
                    if miss in missing_textual:
                       # Usa la media globale denormalizzata invece di zeri
                       output = (visual_global_mean * (visual_max - visual_min)) + visual_min
                       np.save(os.path.join(output_path, f'{miss}.npy'), output.detach().cpu().numpy())
                    else:
                       textual_input = torch.from_numpy(np.load(os.path.join(textual_folder, f'{miss}.npy')))
                       textual_input = (textual_input - textual_min.cpu()) / (textual_max.cpu() - textual_min.cpu() + 1e-8)
                       textual_input = textual_input.to(self.device)
                       output, _, _ = model_impute_visual(textual_input)  # <<< MODIFICATO
                       output = (output * (visual_max - visual_min)) + visual_min
                       np.save(os.path.join(output_path, f'{miss}.npy'), output.detach().cpu().numpy())

    def impute(self): 
        
        self.set_seed()

        output_visual = self.output_paths[0] + '/varautoenc'
        os.makedirs(output_visual, exist_ok=True)
        output_textual = self.output_paths[1] + '/varautoenc'
        os.makedirs(output_textual, exist_ok=True)       

        if self.missing_percentage_visual is not None:
            errors = {'visual':[], 'textual':[]}
            for percentage_visual, percentage_textual in zip(self.missing_percentage_visual, self.missing_percentage_textual):
              try:
                # Crea missing items artificiali e folder
                missing_visual = self.compute_artificial_missing_items(self.dataset_path, percentage_visual, self.items_file, self.seed, file_name=f"artificial_missing_visual_file_{percentage_visual}_varautoenc")  #funzione in simple_imputation
                result_path_percentage_visual = os.path.join(output_visual, f"{percentage_visual}")
                os.makedirs(result_path_percentage_visual, exist_ok=True)   
                missing_visual_npy = [f'{item}.npy' for item in missing_visual]
                visual_folder = f"{self.dataset_path}/non_masked_visual_folder_{percentage_visual}"
                os.makedirs(visual_folder, exist_ok=True)
                for file in os.listdir(self.input_paths[0]):
                    if file.endswith('.npy') and (file not in missing_visual_npy): 
                       np.save(f'{visual_folder}/{file}', np.load(os.path.join(self.input_paths[0], file)))
                
                missing_textual = self.compute_artificial_missing_items(self.dataset_path, percentage_textual, self.items_file, self.seed+1, file_name=f"artificial_missing_textual_file_{percentage_textual}_varautoenc")  #funzione in simple_imputation
                result_path_percentage_textual = os.path.join(output_textual, f"{percentage_textual}")
                os.makedirs(result_path_percentage_textual, exist_ok=True)    
                missing_textual_npy = [f'{item}.npy' for item in missing_textual]
                textual_folder = f"{self.dataset_path}/non_masked_textual_folder_{percentage_textual}"
                os.makedirs(textual_folder, exist_ok=True)
                for file in os.listdir(self.input_paths[1]):
                    if file.endswith('.npy') and (file not in missing_textual_npy): 
                       np.save(f'{textual_folder}/{file}', np.load(os.path.join(self.input_paths[1], file)))

                 # Carica features
                visual_features, textual_features, visual_shape, textual_shape = self.load_features(visual_folder, textual_folder)
                
                # Normalizza
                (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean) = self.normalize_features(visual_features, textual_features)
                
                # Addestra modelli
                model_impute_textual = self._train_varautoencoder_textual(visual_normalized, textual_normalized)
                model_impute_visual = self._train_varautoencoder_visual(textual_normalized, visual_normalized)
                
                # Imputa
                self._impute_textual_modality(missing_textual, missing_visual, visual_folder, model_impute_textual, visual_min, visual_max, textual_min, textual_max, textual_global_mean, result_path_percentage_textual)
                self._impute_visual_modality(missing_visual, missing_textual, textual_folder, model_impute_visual, textual_min, textual_max, visual_min, visual_max, visual_global_mean, result_path_percentage_visual)
             
                # Calcola errore di ricostruzione 
                real_embeddings_visual = [np.load(os.path.join(self.input_paths[0], file)) for file in os.listdir(self.input_paths[0]) if file.endswith('.npy') and file in missing_visual_npy]
                imputed_embeddings_visual = [np.load(os.path.join(result_path_percentage_visual, file)) for file in os.listdir(result_path_percentage_visual) if file.endswith('.npy') and file in missing_visual_npy]
                mse_visual, cosine_sim_visual = self.compute_reconstruction_errors(real_embeddings_visual, imputed_embeddings_visual)
                errors['visual'].append((percentage_visual, mse_visual, cosine_sim_visual))
                
                real_embeddings_textual = [np.load(os.path.join(self.input_paths[1], file)) for file in os.listdir(self.input_paths[1]) if file.endswith('.npy') and file in missing_textual_npy]
                imputed_embeddings_textual = [np.load(os.path.join(result_path_percentage_textual, file)) for file in os.listdir(result_path_percentage_textual) if file.endswith('.npy') and file in missing_textual_npy]
                mse_textual, cosine_sim_textual = self.compute_reconstruction_errors(real_embeddings_textual, imputed_embeddings_textual)
                errors['textual'].append((percentage_textual, mse_textual, cosine_sim_textual))
            
              finally: 
                    self.cleanup_temp_resources([visual_folder, textual_folder])
                    
            return errors

        else:
            visual_folder = self.input_paths[0]
            missing_visual_file_path = os.path.join(self.dataset_path, self.missing_visual_file)
            missing_visual = set(pd.read_csv(missing_visual_file_path, sep='\t', header=None)[0].to_list())
       
            textual_folder = self.input_paths[1]
            missing_textual_file_path = os.path.join(self.dataset_path, self.missing_textual_file)
            missing_textual = set(pd.read_csv(missing_textual_file_path, sep='\t', header=None)[0].to_list())
    
            # Carica features
            visual_features, textual_features, visual_shape, textual_shape = self.load_features(visual_folder, textual_folder)
            
            # Normalizza
            (visual_normalized, textual_normalized, visual_min, visual_max, textual_min, textual_max, visual_global_mean, textual_global_mean) = self.normalize_features(visual_features, textual_features)
            
            # Addestra modelli
            model_impute_textual = self._train_varautoencoder_textual(visual_normalized, textual_normalized)
            model_impute_visual = self._train_varautoencoder_visual(textual_normalized, visual_normalized)
            
            # Imputa
            self._impute_textual_modality(missing_textual, missing_visual, visual_folder, model_impute_textual, visual_min, visual_max, textual_min, textual_max, textual_global_mean, output_textual)
            self._impute_visual_modality(missing_visual, missing_textual, textual_folder, model_impute_visual, textual_min, textual_max, visual_min, visual_max, visual_global_mean, output_visual)
    