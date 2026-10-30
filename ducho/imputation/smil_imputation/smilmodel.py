import torch
import torch.nn as nn
import torch.nn.functional as F
from ducho.imputation.smil_imputation.networks import *

# Loss Functions
# Knowledge Distillation Feature Loss: combine classification loss and feature distillation loss
class KDFeatureLoss(nn.Module):  

    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        super(KDFeatureLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha  # weight for classification loss
        self.beta = beta    # weight for feature distillation loss
    
    def forward(self, features_teacher, features_student, pred_student, label):
        loss_ce = self.cross_entropy(pred_student, label)                       # pred_student: predictions from the student model
        loss_dist = self.l2_loss(features_teacher, features_student)            # features_teacher: features from the teacher model (mode='two') - features_student: features from the student model (mode='one')
        loss = self.alpha * loss_ce + self.beta * loss_dist
        return loss

# Loss to reconstruct missing features
class ReconstructionLoss(nn.Module): 
  
    def __init__(self, reduction='mean'):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, reconstructed, target):
        return self.mse_loss(reconstructed, target)  # reconstructed: reconstructed features - target: original features


# SMIL Model for the imputation of missing modalities
# SMIL (Severely Missing Modality Learning): uses Bayesian meta-learning to perturb the latent space so that the embedding of a single modality can approximate that of the full modality.
class SMILModel:
    
    def __init__(self, visual_dim, textual_dim, gpu_ids, alpha, beta):
        self.visual_dim = visual_dim  
        self.textual_dim = textual_dim  
        self.gpu_ids = gpu_ids  
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
                
        # Create the networks     
        # Inference Network (generate perturbations)
        self.encoder = create_infernet(visual_dim, textual_dim, gpu_ids) 
        # Modality Fuser (merges modalities and applies perturbations)
        self.fuser = create_fuser(visual_dim, textual_dim, num_classes=None, gpu_ids=[])  
        # Loss functions ans move to device
        self.feature_loss = KDFeatureLoss(alpha, beta) 
        self.recon_loss = ReconstructionLoss()
        self.feature_loss = self.feature_loss.to(self.device)
        self.recon_loss = self.recon_loss.to(self.device)
        # Statistics for the averages of modalities
        self.visual_mean = None
        self.textual_mean = None
    
    def setup_optimizers(self, lr=0.001):
        # Optimizer for the encoder
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr) 
        # Optimizer for the fuser
        self.optimizer_fuser = torch.optim.Adam(self.fuser.parameters(), lr=lr) 

    # Calculate the averages of the modalities
    def compute_modality_means(self, visual_features, textual_features):  
        self.visual_mean = torch.mean(visual_features, dim=0)             # visual_features: tensor [N, visual_dim]
        self.textual_mean = torch.mean(textual_features, dim=0)           # textual_features: tensor [N, textual_dim]
     
    def cosine_loss(self, pred, target):
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        return 1 - (pred_norm * target_norm).sum(dim=-1).mean()

    def train_step(self, visual_batch, textual_batch, missing_modality):

        visual_batch = visual_batch.to(self.device)
        textual_batch = textual_batch.to(self.device)
        self.optimizer_encoder.zero_grad()
        self.optimizer_fuser.zero_grad()

        output_teacher, features_teacher, _, _ = self.fuser(
            visual_batch, textual_batch, encoder=None, meta_train=False, mode='two')

        if missing_modality == 'textual':
            output_student, features_student, visual_rec, textual_rec = self.fuser(
                visual_batch, textual_feature=None, encoder=self.encoder,
                textual_mean=self.textual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'],
                meta_train=True, mode='one', missing_modality='textual')
            loss_distill = self.recon_loss(features_student[-1], features_teacher[-1].detach())
            loss_mse = self.recon_loss(textual_rec, textual_batch)
            loss_cos = self.cosine_loss(textual_rec, textual_batch)  # cosine loss
            loss = loss_distill + loss_mse + loss_cos

        elif missing_modality == 'visual':
            output_student, features_student, visual_rec, textual_rec = self.fuser(
                visual_feature=None, textual_feature=textual_batch, encoder=self.encoder,
                visual_mean=self.visual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'],
                meta_train=True, mode='one', missing_modality='visual')
            loss_distill = self.recon_loss(features_student[-1], features_teacher[-1].detach())
            loss_mse = self.recon_loss(visual_rec, visual_batch)
            loss_cos = self.cosine_loss(visual_rec, visual_batch)  # cosine loss
            loss = loss_distill + loss_mse + loss_cos

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.fuser.parameters(), 1.0)
        self.optimizer_encoder.step()
        self.optimizer_fuser.step()
        return loss.item()
    

    # Impute textual features from visual features
    def impute_textual(self, visual_feature):  
        if visual_feature.dim() == 1:                      # visual feature [1, visual_dim] or [visual_dim]
            visual_feature = visual_feature.unsqueeze(0)
        visual_feature = visual_feature.to(self.device)
        self.encoder.eval()
        self.fuser.eval()
        with torch.no_grad(): 
            # Use mode='one' with meta_train=False (no noise during imputation)
            #output, features = self.fuser(visual_feature, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=False, mode='one') 
             _, _, _, textual_reconstructed = self.fuser(visual_feature, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=False, mode='one', missing_modality='textual')
            # Extract the textual component from the concatenated feature, features[0] is [batch, visual_dim + textual_dim], we take only the textual part
            #concat_feature = features[0]                            # [1, 2816]
            #textual_imputed = concat_feature[:, self.visual_dim:]   # [1, 768]
        #return textual_imputed.squeeze(0)                           # [768] - imputed textual feature [textual_dim]
        return textual_reconstructed.squeeze(0)                      # [textual_dim]

    # Impute visual features from textual features
    def impute_visual(self, textual_feature):
        if textual_feature.dim() == 1:                       # textual feature [1, textual_dim] or [textual_dim]
            textual_feature = textual_feature.unsqueeze(0)
        textual_feature = textual_feature.to(self.device)
        self.encoder.eval()
        self.fuser.eval()
        with torch.no_grad():  
            # Use mode='one' with meta_train=False (no noise during imputation)
            #output, features = self.fuser(visual_feature=None, textual_feature=textual_feature, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=False, mode='one', missing_modality='visual')      
            _, _, visual_reconstructed, _ = self.fuser(visual_feature=None, textual_feature=textual_feature, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=False, mode='one', missing_modality='visual')
            # Extract the visual component from the concatenated feature, features[0] is [batch, visual_dim + textual_dim], we take only the visual part
            #concat_feature = features[0]                            # [1, 2816]
            #visual_imputed = concat_feature[:, :self.visual_dim]    # [1, 2048]
        #return visual_imputed.squeeze(0)                            # [2048] - imputed visual feature [visual_dim]
        return visual_reconstructed.squeeze(0)                       # [visual_dim]

    # Set networks to training mode
    def train(self):  

        self.encoder.train() 
        self.fuser.train()

    # Set networks to evaluation mode
    def eval(self):
         
        self.encoder.eval()
        self.fuser.eval()