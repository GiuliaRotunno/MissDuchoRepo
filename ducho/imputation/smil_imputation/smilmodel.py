# SMIL Model Adapted for Feature Vectors, core del modello SMIL per meta-learning Bayesiano
import torch
import torch.nn as nn
import torch.nn.functional as F
from ducho.imputation.smil_imputation.networks import *

# Loss Functions
class KDFeatureLoss(nn.Module):  #Knowledge Distillation Feature Loss: combina classification loss e feature distillation loss

    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        super(KDFeatureLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha  # peso per classification loss
        self.beta = beta    # peso per feature distillation loss
    
    def forward(self, features_teacher, features_student, pred_student, label):
        loss_ce = self.cross_entropy(pred_student, label)    #pred_student: predizioni dal modello student
        loss_dist = self.l2_loss(features_teacher, features_student)  #features_teacher: features dal modello teacher (mode='two')   #features_student: features dal modello student (mode='one')
        loss = self.alpha * loss_ce + self.beta * loss_dist
        return loss

class ReconstructionLoss(nn.Module): #Loss per ricostruire features mancanti
  
    def __init__(self, reduction='mean'):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, reconstructed, target):
        return self.mse_loss(reconstructed, target)  #reconstructed: features ricostruite   #target: features originali

# SMIL Model per imputation di modalità mancanti
# Usa Bayesian meta-learning per perturbare lo spazio latente in modo che l'embedding di una singola modalità possa approssimare quello della modalità completa.
class SMILModel:
    
    def __init__(self, visual_dim, textual_dim, gpu_ids, alpha, beta):
        self.visual_dim = visual_dim  #dimensione visual features (default: 2048 ResNet50)
        self.textual_dim = textual_dim  #dimensione textual features (default: 768 MPNet)
        self.gpu_ids = gpu_ids  #gpu_ids: lista GPU IDs (es. [0] per cuda:0, [] per CPU)
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
                
        # Crea le reti     
        self.encoder = create_infernet(visual_dim, textual_dim, gpu_ids) # Inference Network (genera perturbazioni)
        self.fuser = create_fuser(visual_dim, textual_dim, num_classes=None, gpu_ids=[])  # Modality Fuser (fonde modalità e applica perturbazioni)
        # Loss functions e sposta su device
        self.feature_loss = KDFeatureLoss(alpha, beta) 
        self.recon_loss = ReconstructionLoss()
        self.feature_loss = self.feature_loss.to(self.device)
        self.recon_loss = self.recon_loss.to(self.device)
        # Statistiche per le medie delle modalità
        self.visual_mean = None
        self.textual_mean = None
    
    def setup_optimizers(self, lr=0.001):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr) # Optimizer per encoder
        self.optimizer_fuser = torch.optim.Adam(self.fuser.parameters(), lr=lr) # Optimizer per fuser
    
    def compute_modality_means(self, visual_features, textual_features):  #Calcola le medie delle modalità
        self.visual_mean = torch.mean(visual_features, dim=0)  # visual_features: tensor [N, visual_dim]
        self.textual_mean = torch.mean(textual_features, dim=0)  # textual_features: tensor [N, textual_dim]
     
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
            loss_cos = self.cosine_loss(textual_rec, textual_batch)  # ← cosine loss
            loss = loss_distill + loss_mse + loss_cos

        elif missing_modality == 'visual':
            output_student, features_student, visual_rec, textual_rec = self.fuser(
                visual_feature=None, textual_feature=textual_batch, encoder=self.encoder,
                visual_mean=self.visual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'],
                meta_train=True, mode='one', missing_modality='visual')
            loss_distill = self.recon_loss(features_student[-1], features_teacher[-1].detach())
            loss_mse = self.recon_loss(visual_rec, visual_batch)
            loss_cos = self.cosine_loss(visual_rec, visual_batch)  # ← cosine loss
            loss = loss_distill + loss_mse + loss_cos

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.fuser.parameters(), 1.0)
        self.optimizer_encoder.step()
        self.optimizer_fuser.step()
        return loss.item()
    
    '''backup
    def train_step(self, visual_batch, textual_batch, missing_modality):  # Single training step con meta-learning
        visual_batch = visual_batch.to(self.device)  #batch visual features [batch, visual_dim]
        textual_batch = textual_batch.to(self.device)  #batch textual features [batch, textual_dim]
        # Zero gradients
        self.optimizer_encoder.zero_grad()
        self.optimizer_fuser.zero_grad()
        # Forward pass con entrambe le modalità (teacher)
        output_teacher, features_teacher, _, _ = self.fuser(visual_batch, textual_batch, encoder=None, meta_train=False, mode='two')
       
        # Forward pass con modalità mancante + perturbazioni (student)
        if missing_modality == 'textual':  # Simula textual mancante, usa visual + textual_mean + perturbazioni
            #output_student, features_student = self.fuser(visual_batch, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=True, mode='one', missing_modality='textual')
            output_student, features_student, visual_rec, textual_rec = self.fuser(visual_batch, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=True, mode='one', missing_modality='textual')
            # Loss distillation + loss diretta sulla textual
            loss_distill = self.recon_loss(features_student[-1], features_teacher[-1].detach())
            loss_dec = self.recon_loss(textual_rec, textual_batch)  #supervisione diretta
            loss = loss_distill + loss_dec
        
        elif missing_modality == 'visual':  # Simula visual mancante, usa textual + visual_mean + perturbazioni
            #output_student, features_student = self.fuser(visual_feature=None, textual_feature=textual_batch, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=True, mode='one', missing_modality='visual')
            output_student, features_student, visual_rec, textual_rec = self.fuser(visual_feature=None, textual_feature=textual_batch, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=True, mode='one', missing_modality='visual')
            # Loss distillation + loss diretta sulla visual
            loss_distill = self.recon_loss(features_student[-1], features_teacher[-1].detach())
            loss_dec = self.recon_loss(visual_rec, visual_batch)  #supervisione diretta
            loss = loss_distill + loss_dec

        else:
            raise ValueError(f"missing_modality must be 'visual' or 'textual', got {missing_modality}")
        
        # Calcola loss: distillation tra teacher e student
        #loss = self.recon_loss(features_student[-1], features_teacher[-1].detach())  # Usiamo l'ultima feature layer per la distillation
        # Backward
        loss.backward()
        # Update
        self.optimizer_encoder.step()
        self.optimizer_fuser.step()
        return loss.item()
    '''

    def impute_textual(self, visual_feature):  #Imputa textual feature da visual feature
        if visual_feature.dim() == 1:  #visual feature [1, visual_dim] o [visual_dim]
            visual_feature = visual_feature.unsqueeze(0)
        visual_feature = visual_feature.to(self.device)
        self.encoder.eval()
        self.fuser.eval()
        with torch.no_grad(): # Usa mode='one' con meta_train=False (no rumore durante imputation)
            #output, features = self.fuser(visual_feature, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=False, mode='one') 
             _, _, _, textual_reconstructed = self.fuser(visual_feature, textual_feature=None, encoder=self.encoder, textual_mean=self.textual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=False, mode='one', missing_modality='textual')
            # Estrai la componente textual dalla feature concatenata, features[0] è [batch, visual_dim + textual_dim], prendiamo solo la parte textual
            #concat_feature = features[0]  # [1, 2816]
            #textual_imputed = concat_feature[:, self.visual_dim:]  # [1, 768]
        #return textual_imputed.squeeze(0)  # [768]    #textual feature imputata [textual_dim]
        return textual_reconstructed.squeeze(0)  # [textual_dim]

    
    def impute_visual(self, textual_feature): #Imputa visual feature da textual feature
        if textual_feature.dim() == 1:   #textual feature [1, textual_dim] o [textual_dim]
            textual_feature = textual_feature.unsqueeze(0)
        textual_feature = textual_feature.to(self.device)
        self.encoder.eval()
        self.fuser.eval()
        with torch.no_grad():  # Usa mode='one' con meta_train=False (no rumore durante imputation)
            #output, features = self.fuser(visual_feature=None, textual_feature=textual_feature, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=False, mode='one', missing_modality='visual') # NO rumore durante imputation          
            _, _, visual_reconstructed, _ = self.fuser(visual_feature=None, textual_feature=textual_feature, encoder=self.encoder, visual_mean=self.visual_mean, noise_layers=['fc_concat','fc1','fc2','fc3'], meta_train=False, mode='one', missing_modality='visual')
            # Estrai la componente visual dalla feature concatenata, features[0] è [batch, visual_dim + textual_dim], prendiamo solo la parte visual
            #concat_feature = features[0]  # [1, 2816]
            #visual_imputed = concat_feature[:, :self.visual_dim]  #[1, 2048]
        #return visual_imputed.squeeze(0)  # [2048]  #visual feature imputata [visual_dim]
        return visual_reconstructed.squeeze(0)  # [visual_dim]
    
    def train(self):   #Set networks to training mode
        self.encoder.train()
        self.fuser.train()
    
    def eval(self):    #Set networks to evaluation mode
        self.encoder.eval()
        self.fuser.eval()