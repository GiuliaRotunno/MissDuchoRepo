#MMIN Model Adapted for Feature Vectors (https://github.com/AIM3-RUC/MMIN), core del modello MMIN
import torch
import torch.nn as nn
import torch.nn.functional as F
from ducho.imputation.mmin_imputation.networks import *

# MMIN Model per imputazione di modalità mancanti
# MMIN (Missing Modality Imagination Network) impara a ricostruire le modalità mancanti attraverso un autoencoder residuale.
# Durante training: simula missing modalities random, l'autoencoder impara a ricostruire le features complete, usa cycle consistency per stabilità
class MMINModel:  
    def __init__(self, visual_dim=2048, textual_dim=768, embd_size_v=128, embd_size_t=128, AE_layers=[128, 64, 32], n_blocks=3, dropout=0.3, gpu_ids=[0]):
        self.visual_dim = visual_dim    #visual_dim: dimensione visual features (default: 2048 ResNet50)
        self.textual_dim = textual_dim  #textual_dim: dimensione textual features (default: 768 MPNet)
        self.embd_size_v = embd_size_v  #embd_size_v: dimensione embedding visual (default: 128)
        self.embd_size_t = embd_size_t  #embd_size_t: dimensione embedding textual (default: 128)
        self.gpu_ids = gpu_ids   #gpu_ids: lista GPU IDs
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
        # Modality encoders (riducono dim originale a embd_size)
        self.netV = create_modality_encoder(visual_dim, embd_size_v, dropout=dropout, gpu_ids=gpu_ids)        
        self.netT = create_modality_encoder(textual_dim, embd_size_t, dropout=dropout, gpu_ids=gpu_ids)
        # Modality decoders (riportano da embd_size a dim originale)
        self.netV_dec = create_modality_decoder(embd_size_v, visual_dim, dropout=dropout, gpu_ids=gpu_ids)
        self.netT_dec = create_modality_decoder(embd_size_t, textual_dim, dropout=dropout, gpu_ids=gpu_ids)
        # Autoencoder
        AE_input_dim = embd_size_v + embd_size_t 
        self.netAE = create_residual_ae(AE_input_dim, AE_layers, n_blocks, dropout=0.0, use_bn=False, gpu_ids=gpu_ids)  #AE_layers: layers dell'autoencoder  #n_blocks: numero blocchi residuali
        # Cycle AE
        self.netAE_cycle = create_residual_ae(AE_input_dim, AE_layers, n_blocks, dropout=0.0, use_bn=False, gpu_ids=gpu_ids)
        # Loss functions
        self.criterion_mse = nn.MSELoss()
        self.criterion_mse = self.criterion_mse.to(self.device)
    
    def setup_optimizers(self, lr=0.0002):
        parameters = list(self.netV.parameters()) + list(self.netT.parameters()) + \
                     list(self.netV_dec.parameters()) + list(self.netT_dec.parameters()) + \
                     list(self.netAE.parameters()) + list(self.netAE_cycle.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))
    
    def set_input(self, visual, textual, missing_modality=None):   # Set input data con simulazione di missing modalities
        visual = visual.to(self.device)     # visual: visual features [batch, visual_dim]
        textual = textual.to(self.device)   #textual: textual features [batch, textual_dim]
        if missing_modality == 'visual':    # Simula visual mancante (maschera a zero)
            self.visual_miss = torch.zeros_like(visual)
            self.textual_miss = textual
            self.visual_available = visual  # per cycle loss
            self.textual_available = textual
        elif missing_modality == 'textual': # Simula textual mancante (maschera a zero)
            self.visual_miss = visual
            self.textual_miss = torch.zeros_like(textual)
            self.visual_available = visual
            self.textual_available = textual # per cycle loss
        else:                                # Nessuna missing (per testing)
            self.visual_miss = visual
            self.textual_miss = textual
            self.visual_available = visual
            self.textual_available = textual
    
    def forward(self):  #Forward pass -> Durante training: processa modalità con missing, ricostruisce features complete via AE, applica cycle consistency
        # Encode modality (anche se mancanti/mascherate a zero)
        self.feat_V_miss = self.netV(self.visual_miss)
        self.feat_T_miss = self.netT(self.textual_miss)
        self.feat_fusion_miss = torch.cat([self.feat_V_miss, self.feat_T_miss], dim=-1)  # Concatena features
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)  # Ricostruzione via autoencoder
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)  # Cycle consistency (ricostruisci di nuovo)
        return self.recon_fusion, self.latent
    
    def compute_target(self):  #Calcola target features: durante training, il target sono le features delle modalità disponibili
        with torch.no_grad():
            feat_V_available = self.netV(self.visual_available)
            feat_T_available = self.netT(self.textual_available)
            self.target_fusion = torch.cat([feat_V_available, feat_T_available], dim=-1)
    
    ''' backup
    def backward(self, mse_weight=1.0, cycle_weight=1.0):  #Calculate loss and backward
        # Loss 1: Ricostruzione deve matchare target
        self.loss_mse = mse_weight * self.criterion_mse(self.recon_fusion, self.target_fusion)  #mse_weight: peso per reconstruction loss
        # Loss 2: Cycle consistency (ricostruisci → ricostruisci → deve tornare all'input)
        self.loss_cycle = cycle_weight * self.criterion_mse(self.recon_cycle, self.feat_fusion_miss.detach())  #cycle_weight: peso per cycle consistency loss
        # Total loss
        loss = self.loss_mse + self.loss_cycle
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.netV.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netT.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE_cycle.parameters(), 1.0)
        return {'loss_mse': self.loss_mse.item(), 'loss_cycle': self.loss_cycle.item(), 'loss_total': loss.item()} #dizionario con le loss
    '''

    def backward(self, mse_weight, cycle_weight, visual_available, textual_available, missing_modality):
        # Loss esistente: ricostruzione nello spazio embedding
        self.loss_mse = mse_weight * self.criterion_mse(self.recon_fusion, self.target_fusion)
        self.loss_cycle = cycle_weight * self.criterion_mse(self.recon_cycle, self.feat_fusion_miss.detach())
        # NUOVO: loss diretta sul decoder nello spazio originale
        if missing_modality == 'textual':
            textual_embd = self.recon_fusion[:, self.embd_size_v:]  # [batch, 128]
            textual_decoded = self.netT_dec(textual_embd)           # [batch, 768]
            self.loss_dec = self.criterion_mse(textual_decoded, textual_available)
        elif missing_modality == 'visual':
            visual_embd = self.recon_fusion[:, :self.embd_size_v]   # [batch, 128]
            visual_decoded = self.netV_dec(visual_embd)              # [batch, 2048]
            self.loss_dec = self.criterion_mse(visual_decoded, visual_available)
        
        loss = self.loss_mse + self.loss_cycle + self.loss_dec
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.netV.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netT.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE_cycle.parameters(), 1.0)
        return {'loss_mse': self.loss_mse.item(), 'loss_cycle': self.loss_cycle.item(), 'loss_dec': self.loss_dec.item(), 'loss_total': loss.item()}

    '''backup
    def optimize_parameters(self, mse_weight, cycle_weight, missing_modality):   #missing_modality: quale modalità simulare come mancante
        # Forward pass
        self.forward()
        # Compute target
        self.compute_target()
        # Backward
        self.optimizer.zero_grad()
        loss_dict = self.backward(mse_weight, cycle_weight) 
        self.optimizer.step()
        return loss_dict   #dizionario con le loss
    '''
    
    def optimize_parameters(self, mse_weight, cycle_weight, missing_modality='textual'):
        self.forward()
        self.compute_target()
        self.optimizer.zero_grad()
        # passa i dati originali per la loss sul decoder
        loss_dict = self.backward(mse_weight, cycle_weight,
            visual_available=self.visual_available,
            textual_available=self.textual_available,
            missing_modality=missing_modality
        )
        self.optimizer.step()
        return loss_dict

    ''' backup
    def impute_textual(self, visual_feature):  #Imputa textual feature da visual feature
        if visual_feature.dim() == 1:  # Assicura che sia [1, visual_dim]
            visual_feature = visual_feature.unsqueeze(0)  #visual_feature: visual feature [1, visual_dim] o [visual_dim]
        visual_feature = visual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netT_dec.eval()
        self.netAE.eval()
        with torch.no_grad():
            # Visual disponibile, textual mancante (zero)
            feat_v = self.netV(visual_feature)
            # Assicura che feat_v sia 2D [batch, embd_size_v]
            if feat_v.dim() == 3:
                feat_v = feat_v.squeeze(1)  # Rimuovi dimensione extra
            feat_t_zero = torch.zeros(visual_feature.size(0), self.embd_size_t, device=self.device)
            # Concatena
            feat_fusion = torch.cat([feat_v, feat_t_zero], dim=-1)
            # Ricostruisci
            recon_fusion, _ = self.netAE(feat_fusion)
            # Estrai componente textual dalla ricostruzione  #recon_fusion = [batch, embd_size_v + embd_size_t] -> componente textual: recon_fusion[:, embd_size_v:]
            textual_embd_imputed = recon_fusion[:, self.embd_size_v:]
            # Decodifica a dimensione originale
            textual_imputed_full = self.netT_dec(textual_embd_imputed)  # [batch, textual_dim]
        return textual_imputed_full.squeeze(0)   #textual_imputed_full: textual feature imputata [textual_dim]
    '''
    
    def impute_textual(self, visual_feature):
        if visual_feature.dim() == 1:  # Assicura che sia [1, visual_dim]
            visual_feature = visual_feature.unsqueeze(0)  #visual_feature: visual feature [1, visual_dim] o [visual_dim]
        visual_feature = visual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netT_dec.eval()
        self.netAE.eval()   

        with torch.no_grad():
            feat_v = self.netV(visual_feature)
            # CORRETTO: usa netT(zeros) come durante il training, non zeros diretti
            zeros_textual = torch.zeros(visual_feature.size(0), self.textual_dim, device=self.device)
            feat_t_from_zeros = self.netT(zeros_textual)  # [1, 128] — coerente col training
            feat_fusion = torch.cat([feat_v, feat_t_from_zeros], dim=-1)
            recon_fusion, _ = self.netAE(feat_fusion)
            textual_embd = recon_fusion[:, self.embd_size_v:]
            textual_imputed = self.netT_dec(textual_embd)
        return textual_imputed.squeeze(0)
    
    '''backup
    def impute_visual(self, textual_feature):  #Imputa visual feature da textual feature
        if textual_feature.dim() == 1:  # Assicurati che sia [1, textual_dim]
            textual_feature = textual_feature.unsqueeze(0) #textual_feature: textual feature [1, textual_dim] o [textual_dim]
        textual_feature = textual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netV_dec.eval()
        self.netAE.eval()
        with torch.no_grad():
            # Textual disponibile, visual mancante (zero)
            feat_t = self.netT(textual_feature)
            # Assicura che feat_t sia 2D [batch, embd_size_t]
            if feat_t.dim() == 3:
                feat_t = feat_t.squeeze(1)  # Rimuovi dimensione extra
            feat_v_zero = torch.zeros(textual_feature.size(0), self.embd_size_v, device=self.device)            
            # Concatena
            feat_fusion = torch.cat([feat_v_zero, feat_t], dim=-1)
            # Ricostruisci
            recon_fusion, _ = self.netAE(feat_fusion)
            # Estrai componente visual dalla ricostruzione  # recon_fusion = [batch, embd_size_v + embd_size_t] --> componente visual: recon_fusion[:, :embd_size_v]
            visual_embd_imputed = recon_fusion[:, :self.embd_size_v]
            # Decodifica a dimensione originale
            visual_imputed_full = self.netV_dec(visual_embd_imputed)  # [batch, visual_dim]
        return visual_imputed_full.squeeze(0)   # visual_imputed_full: visual feature imputata [visual_dim]
    '''

    def impute_visual(self, textual_feature):
        if textual_feature.dim() == 1:  # Assicurati che sia [1, textual_dim]
            textual_feature = textual_feature.unsqueeze(0) #textual_feature: textual feature [1, textual_dim] o [textual_dim]
        textual_feature = textual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netV_dec.eval()
        self.netAE.eval()
 
        with torch.no_grad():
            feat_t = self.netT(textual_feature)
            zeros_visual = torch.zeros(textual_feature.size(0), self.visual_dim, device=self.device)
            feat_v_from_zeros = self.netV(zeros_visual)  # coerente col training
            feat_fusion = torch.cat([feat_v_from_zeros, feat_t], dim=-1)
            recon_fusion, _ = self.netAE(feat_fusion)
            visual_embd = recon_fusion[:, :self.embd_size_v]
            visual_imputed = self.netV_dec(visual_embd)
        return visual_imputed.squeeze(0)

    def train(self):  #Set networks to training mode
        self.netV.train()
        self.netT.train() 
        self.netV_dec.train()
        self.netT_dec.train()
        self.netAE.train() 
        self.netAE_cycle.train()
    
    def eval(self):   #Set networks to evaluation mode
        self.netV.eval()
        self.netT.eval()
        self.netV_dec.eval()
        self.netT_dec.eval()
        self.netAE.eval()
        self.netAE_cycle.eval()