import torch
import torch.nn as nn
import torch.nn.functional as F
from ducho.imputation.mmin_imputation.networks import *
 
# MMIN Model for the imputation of missing modalities
# MMIN (Missing Modality Imagination Network) learn to reconstruct missing modalities through a residual autoencoder.
# During training: simulates random missing modalities, the autoencoder learns to reconstruct complete features, uses cycle consistency for stability
class MMINModel:  
    def __init__(self, visual_dim=2048, textual_dim=768, embd_size_v=128, embd_size_t=128, AE_layers=[128, 64, 32], n_blocks=3, dropout=0.3, gpu_ids=[0]):
        self.visual_dim = visual_dim    
        self.textual_dim = textual_dim  
        self.embd_size_v = embd_size_v  
        self.embd_size_t = embd_size_t  
        self.gpu_ids = gpu_ids   
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')

        # Modality encoders (reduce original dim to embd_size)
        self.netV = create_modality_encoder(visual_dim, embd_size_v, dropout=dropout, gpu_ids=gpu_ids)        
        self.netT = create_modality_encoder(textual_dim, embd_size_t, dropout=dropout, gpu_ids=gpu_ids)
        # Modality decoders (reverts embd_size to original dim)
        self.netV_dec = create_modality_decoder(embd_size_v, visual_dim, dropout=dropout, gpu_ids=gpu_ids)
        self.netT_dec = create_modality_decoder(embd_size_t, textual_dim, dropout=dropout, gpu_ids=gpu_ids)
        # Autoencoder
        AE_input_dim = embd_size_v + embd_size_t 
        self.netAE = create_residual_ae(AE_input_dim, AE_layers, n_blocks, dropout=0.0, use_bn=False, gpu_ids=gpu_ids)   # n_blocks: number of residual blocks
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

    # Set input data with simulation of missing modalities
    def set_input(self, visual, textual, missing_modality=None):
           
        visual = visual.to(self.device)      # visual: visual features [batch, visual_dim]
        textual = textual.to(self.device)    # textual: textual features [batch, textual_dim]

        if missing_modality == 'visual':     # Simulate missing visual (zero mask)
            self.visual_miss = torch.zeros_like(visual)
            self.textual_miss = textual
            self.visual_available = visual   # for cycle loss
            self.textual_available = textual
        elif missing_modality == 'textual':  # Simulate missing textual (zero mask)
            self.visual_miss = visual
            self.textual_miss = torch.zeros_like(textual)
            self.visual_available = visual
            self.textual_available = textual # for cycle loss
        else:                                # None missing (for testing)
            self.visual_miss = visual
            self.textual_miss = textual
            self.visual_available = visual
            self.textual_available = textual

    # Forward pass -> During training: processes missing modalities, reconstructs complete features via AE, applies cycle consistency
    def forward(self): 
        # Encode modality (even if missing/masked to zero)
        self.feat_V_miss = self.netV(self.visual_miss)
        self.feat_T_miss = self.netT(self.textual_miss)
        # Concatenate features
        self.feat_fusion_miss = torch.cat([self.feat_V_miss, self.feat_T_miss], dim=-1)  
        # Ricostruct via autoencoder
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)  
        # Cycle consistency
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)  
        return self.recon_fusion, self.latent

    # Calculate target features: during training, the target are the features of the available modalities
    def compute_target(self): 

        with torch.no_grad():
            feat_V_available = self.netV(self.visual_available)
            feat_T_available = self.netT(self.textual_available)
            self.target_fusion = torch.cat([feat_V_available, feat_T_available], dim=-1)
    

    def backward(self, mse_weight, cycle_weight, visual_available, textual_available, missing_modality):
        # Existing loss: reconstruction in the embedding space
        self.loss_mse = mse_weight * self.criterion_mse(self.recon_fusion, self.target_fusion)
        self.loss_cycle = cycle_weight * self.criterion_mse(self.recon_cycle, self.feat_fusion_miss.detach())
        # Loss on the decoder in the original space
        if missing_modality == 'textual':
            textual_embd = self.recon_fusion[:, self.embd_size_v:]  # [batch, 128]
            textual_decoded = self.netT_dec(textual_embd)           # [batch, 768]
            self.loss_dec = self.criterion_mse(textual_decoded, textual_available)
        elif missing_modality == 'visual':
            visual_embd = self.recon_fusion[:, :self.embd_size_v]   # [batch, 128]
            visual_decoded = self.netV_dec(visual_embd)             # [batch, 2048]
            self.loss_dec = self.criterion_mse(visual_decoded, visual_available)
        
        loss = self.loss_mse + self.loss_cycle + self.loss_dec
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.netV.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netT.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netAE_cycle.parameters(), 1.0)

        return {'loss_mse': self.loss_mse.item(), 'loss_cycle': self.loss_cycle.item(), 'loss_dec': self.loss_dec.item(), 'loss_total': loss.item()}

    
    def optimize_parameters(self, mse_weight, cycle_weight, missing_modality='textual'):

        self.forward()
        self.compute_target()
        self.optimizer.zero_grad()
        # passes the original data for loss to the decoder
        loss_dict = self.backward(mse_weight, cycle_weight,
            visual_available=self.visual_available,
            textual_available=self.textual_available,
            missing_modality=missing_modality
        )
        self.optimizer.step()

        return loss_dict


    def impute_textual(self, visual_feature):

        if visual_feature.dim() == 1:                     # Make sure it is [1, visual_dim]
            visual_feature = visual_feature.unsqueeze(0)  # visual_feature: visual feature [1, visual_dim] or [visual_dim]
        visual_feature = visual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netT_dec.eval()
        self.netAE.eval()   

        with torch.no_grad():
            feat_v = self.netV(visual_feature)
            # Use netT(zeros) like during the training phase
            zeros_textual = torch.zeros(visual_feature.size(0), self.textual_dim, device=self.device)
            feat_t_from_zeros = self.netT(zeros_textual)  # [1, 128]
            feat_fusion = torch.cat([feat_v, feat_t_from_zeros], dim=-1)
            recon_fusion, _ = self.netAE(feat_fusion)
            textual_embd = recon_fusion[:, self.embd_size_v:]
            textual_imputed = self.netT_dec(textual_embd)

        return textual_imputed.squeeze(0)
    

    def impute_visual(self, textual_feature):
        if textual_feature.dim() == 1:                      # Make sure it is [1, textual_dim]
            textual_feature = textual_feature.unsqueeze(0)  # textual_feature: textual feature [1, textual_dim] or [textual_dim]
        textual_feature = textual_feature.to(self.device)
        
        self.netV.eval()
        self.netT.eval()
        self.netV_dec.eval()
        self.netAE.eval()
 
        with torch.no_grad():
            feat_t = self.netT(textual_feature)
            zeros_visual = torch.zeros(textual_feature.size(0), self.visual_dim, device=self.device)
            feat_v_from_zeros = self.netV(zeros_visual)
            feat_fusion = torch.cat([feat_v_from_zeros, feat_t], dim=-1)
            recon_fusion, _ = self.netAE(feat_fusion)
            visual_embd = recon_fusion[:, :self.embd_size_v]
            visual_imputed = self.netV_dec(visual_embd)
        return visual_imputed.squeeze(0)

    # Set networks to training mode
    def train(self): 

        self.netV.train()
        self.netT.train() 
        self.netV_dec.train()
        self.netT_dec.train()
        self.netAE.train() 
        self.netAE_cycle.train()

    # Set networks to evaluation mode
    def eval(self): 
          
        self.netV.eval()
        self.netT.eval()
        self.netV_dec.eval()
        self.netT_dec.eval()
        self.netAE.eval()
        self.netAE_cycle.eval()