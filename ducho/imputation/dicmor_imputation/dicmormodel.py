import torch
import torch.nn as nn
import torch.nn.functional as F
from ducho.imputation.dicmor_imputation.networks import *
from random import sample

class MSE(nn.Module): #MSE Loss 
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse 

# DiCMoR usa normalizing flow per trasferire distribuzioni tra modalità mantenendo consistenza
class DICMORModel:  #DiCMoR Model per imputation di modalità mancanti    
    def __init__(self, visual_dim, textual_dim, embd_size_v, embd_size_t, n_flow=32, n_block=1, rcan_blocks=20, gpu_ids=[0]):
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        self.embd_size_v = embd_size_v
        self.embd_size_t = embd_size_t
        self.gpu_ids = gpu_ids
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')
        
        # Crea le reti       
        # Projection layers (riducono dim originale a embd_size)
        self.proj_v = nn.Conv1d(visual_dim, embd_size_v, kernel_size=1, padding=0, bias=False)
        self.proj_t = nn.Conv1d(textual_dim, embd_size_t, kernel_size=1, padding=0, bias=False)
        # Deprojection layers (riportano da embd_size a dim originale)
        self.deproj_v = nn.Conv1d(embd_size_v, visual_dim, kernel_size=1, padding=0, bias=False)
        self.deproj_t = nn.Conv1d(embd_size_t, textual_dim, kernel_size=1, padding=0, bias=False)
        # Normalizing Flows per ogni modalità 
        self.flow_v = create_glow(embd_size_v, n_flow, n_block, gpu_ids)
        self.flow_t = create_glow(embd_size_t, n_flow, n_block, gpu_ids)
        # RCAN Refinement networks
        self.rec_v = create_rcan_refiner(embd_size_v, rcan_blocks, reduction=16, gpu_ids=gpu_ids)        
        self.rec_t = create_rcan_refiner(embd_size_t, rcan_blocks, reduction=16, gpu_ids=gpu_ids)
        # Concatenation layers (quando abbiamo 2 modalità disponibili)
        self.cat_v = nn.Conv1d(embd_size_v * 2, embd_size_v, kernel_size=1, padding=0)
        self.cat_t = nn.Conv1d(embd_size_t * 2, embd_size_t, kernel_size=1, padding=0)
        # Sposta tutto su device
        self.proj_v = self.proj_v.to(self.device)
        self.proj_t = self.proj_t.to(self.device)
        self.deproj_v = self.deproj_v.to(self.device)
        self.deproj_t = self.deproj_t.to(self.device)
        self.cat_v = self.cat_v.to(self.device)
        self.cat_t = self.cat_t.to(self.device)       
        # Loss functions
        self.criterion_mse = MSE()
    
    def setup_optimizers(self, lr=0.0001):
        parameters = list(self.proj_v.parameters()) + list(self.proj_t.parameters()) + \
                     list(self.deproj_v.parameters()) + list(self.deproj_t.parameters()) + \
                     list(self.flow_v.parameters()) + list(self.flow_t.parameters()) + \
                     list(self.rec_v.parameters()) + list(self.rec_t.parameters()) + \
                     list(self.cat_v.parameters()) + list(self.cat_t.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))
    
    '''backup
    def forward(self, visual, textual, num_modal=None):
        visual = visual.to(self.device)
        textual = textual.to(self.device)
        # Project features
        # Input: [batch, 1, dim] → Transpose → [batch, dim, 1] per Conv1d
        x_v = visual.unsqueeze(-1)  # [batch, 1, visual_dim, 1]
        x_t = textual.unsqueeze(-1)  # [batch, 1, textual_dim, 1]

        x_v = x_v.squeeze(1)
        x_t = x_t.squeeze(1)
        
        with torch.no_grad():
            proj_x_v = self.proj_v(x_v)  # [batch, embd_size, 1]
            proj_x_t = self.proj_t(x_t)  # [batch, embd_size, 1]
        # Salva per reconstruction loss
        conv_feat_v = proj_x_v.clone()
        conv_feat_t = proj_x_t.clone()
        
        # Normalizing flow forward
        # Aggiunge una dimensione per fare [batch, channel, height, width]
        proj_x_v_4d = proj_x_v.unsqueeze(-1)  # [batch, embd_size, 1, 1]
        proj_x_t_4d = proj_x_t.unsqueeze(-1)  # [batch, embd_size, 1, 1]
        log_p_sum_v, logdet_v, z_outs_v = self.flow_v(proj_x_v_4d)
        z_v = z_outs_v  # lista di latent codes
        log_p_sum_t, logdet_t, z_outs_t = self.flow_t(proj_x_t_4d)
        z_t = z_outs_t
        
        # Calculate flow loss
        log_p_v = logdet_v.sum() + log_p_sum_v
        log_p_v = torch.max(torch.zeros_like(log_p_v), (-log_p_v / (np.log(2) * proj_x_v_4d.numel())))
        log_p_t = logdet_t.sum() + log_p_sum_t
        log_p_t = torch.max(torch.zeros_like(log_p_t), (-log_p_t / (np.log(2) * proj_x_t_4d.numel())))
        
        # Random select available modality
        if num_modal is None:
            num_modal = 2  # default: entrambe disponibili
        modal_idx = [0, 1]  # 0: visual, 1: textual
        ava_modal_idx = sample(modal_idx, min(num_modal, 2))
        
        # Ricostruzione basata su modalità disponibili
        if num_modal == 1:  # Una sola modalità disponibile
            if ava_modal_idx[0] == 0:  # Ha visual, manca textual
                # Ricostruisci textual da visual usando reverse flow
                proj_x_t = self.flow_t.reverse(z_v, reconstruct=True).squeeze(-1)
                proj_x_t = self.rec_t(proj_x_t)
                loss_rec = self.criterion_mse(proj_x_t, conv_feat_t.detach())
            else:  # Ha textual, manca visual
                # Ricostruisci visual da textual usando reverse flow
                proj_x_v = self.flow_v.reverse(z_t, reconstruct=True).squeeze(-1)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.criterion_mse(proj_x_v, conv_feat_v.detach())
        
        elif num_modal == 2:  # Entrambe disponibili (no missing)
            loss_rec = torch.tensor(0.0, device=self.device)
        
        else:
            loss_rec = torch.tensor(0.0, device=self.device)
        
        feat_v = proj_x_v.squeeze(-1)  # Rimuovi dimensioni extra per output -> [batch, embd_size]
        feat_t = proj_x_t.squeeze(-1)
        
        return {'feat_v': feat_v,'feat_t': feat_t,'log_p_v': log_p_v,'log_p_t': log_p_t,'loss_rec': loss_rec,'ava_modal_idx': ava_modal_idx}
        '''
    
    def forward(self, visual, textual, num_modal=None):
        visual = visual.to(self.device)
        textual = textual.to(self.device)

        x_v = visual.unsqueeze(-1).squeeze(1)   # [batch, visual_dim, 1]
        x_t = textual.unsqueeze(-1).squeeze(1)  # [batch, textual_dim, 1]

        # RIMOSSO torch.no_grad() — i projection layers devono imparare
        proj_x_v = self.proj_v(x_v)  # [batch, embd_size_v, 1]
        proj_x_t = self.proj_t(x_t)  # [batch, embd_size_t, 1]

        # Salva per reconstruction loss nello spazio embedding
        conv_feat_v = proj_x_v.detach().clone()
        conv_feat_t = proj_x_t.detach().clone()

        # Salva features originali per loss_dec nello spazio originale
        orig_feat_v = x_v  # [batch, visual_dim, 1]
        orig_feat_t = x_t  # [batch, textual_dim, 1]

        # Normalizing flow forward
        proj_x_v_4d = proj_x_v.unsqueeze(-1)
        proj_x_t_4d = proj_x_t.unsqueeze(-1)
        log_p_sum_v, logdet_v, z_outs_v = self.flow_v(proj_x_v_4d)
        z_v = z_outs_v
        log_p_sum_t, logdet_t, z_outs_t = self.flow_t(proj_x_t_4d)
        z_t = z_outs_t

        # Flow loss
        log_p_v = logdet_v.sum() + log_p_sum_v
        log_p_v = torch.max(torch.zeros_like(log_p_v), (-log_p_v / (np.log(2) * proj_x_v_4d.numel())))
        log_p_t = logdet_t.sum() + log_p_sum_t
        log_p_t = torch.max(torch.zeros_like(log_p_t), (-log_p_t / (np.log(2) * proj_x_t_4d.numel())))

        if num_modal is None:
            num_modal = 2
        modal_idx = [0, 1]
        ava_modal_idx = sample(modal_idx, min(num_modal, 2))

        loss_rec = torch.tensor(0.0, device=self.device)
        loss_dec = torch.tensor(0.0, device=self.device)

        if num_modal == 1:
            if ava_modal_idx[0] == 0:  # Ha visual, manca textual
                proj_x_t = self.flow_t.reverse(z_v, reconstruct=True).squeeze(-1)
                proj_x_t = self.rec_t(proj_x_t)
                loss_rec = self.criterion_mse(proj_x_t, conv_feat_t)
                # NUOVO: loss nello spazio originale
                textual_decoded = self.deproj_t(proj_x_t)   # [batch, textual_dim, 1]
                loss_dec = self.criterion_mse(textual_decoded, orig_feat_t)
            else:  # Ha textual, manca visual
                proj_x_v = self.flow_v.reverse(z_t, reconstruct=True).squeeze(-1)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.criterion_mse(proj_x_v, conv_feat_v)
                # NUOVO: loss nello spazio originale
                visual_decoded = self.deproj_v(proj_x_v)    # [batch, visual_dim, 1]
                loss_dec = self.criterion_mse(visual_decoded, orig_feat_v)

        feat_v = proj_x_v.squeeze(-1)
        feat_t = proj_x_t.squeeze(-1)

        return {
            'feat_v': feat_v, 'feat_t': feat_t,
            'log_p_v': log_p_v, 'log_p_t': log_p_t,
            'loss_rec': loss_rec, 'loss_dec': loss_dec,
            'ava_modal_idx': ava_modal_idx
        }


    '''backup
    def backward(self, outputs, log_p_weight=0.1): #Calculate loss and backward
        # Combine losses
        loss_rec = outputs['loss_rec']
        log_p_v = outputs['log_p_v']
        log_p_t = outputs['log_p_t']

        # riduco i tensori non scalari a scalari
        log_p_v_s = log_p_v.mean()
        log_p_t_s = log_p_t.mean()

        total_loss = loss_rec + log_p_weight * (log_p_v_s + log_p_t_s)
        total_loss.backward()
        
        return { 'loss_rec': loss_rec.item() if isinstance(loss_rec, torch.Tensor) else loss_rec,
                 'log_p_v': log_p_v_s.item(),
                 'log_p_t': log_p_t_s.item(),
                 'loss_total': total_loss.item()}
    '''

    def backward(self, outputs, log_p_weight=0.1, dec_weight=1.0):
        loss_rec = outputs['loss_rec']
        loss_dec = outputs['loss_dec']
        log_p_v_s = outputs['log_p_v'].mean()
        log_p_t_s = outputs['log_p_t'].mean()

        total_loss = loss_rec + dec_weight * loss_dec + log_p_weight * (log_p_v_s + log_p_t_s)
        total_loss.backward()

        return {
            'loss_rec': loss_rec.item() if isinstance(loss_rec, torch.Tensor) else loss_rec,
            'loss_dec': loss_dec.item() if isinstance(loss_dec, torch.Tensor) else loss_dec,
            'log_p_v': log_p_v_s.item(),
            'log_p_t': log_p_t_s.item(),
            'loss_total': total_loss.item()
        }
    '''backup
    def optimize_parameters(self, visual_batch, textual_batch, num_modal=1, log_p_weight=0.1):
        # Forward
        outputs = self.forward(visual_batch, textual_batch, num_modal)
        # Debug: stampa le shape dei loss
        loss_rec = outputs['loss_rec']
        log_p_v  = outputs['log_p_v']
        log_p_t  = outputs['log_p_t']
        # Backward
        self.optimizer.zero_grad()
        loss_dict = self.backward(outputs, log_p_weight=log_p_weight)
        # Gradient clipping
        nn.utils.clip_grad_value_([p for p in self.get_all_parameters() if p.requires_grad], 1.0)
        # Update
        self.optimizer.step()
        return loss_dict
    '''

    def optimize_parameters(self, visual_batch, textual_batch, num_modal=1, log_p_weight=0.1, dec_weight=1.0):
        outputs = self.forward(visual_batch, textual_batch, num_modal)
        self.optimizer.zero_grad()
        loss_dict = self.backward(outputs, log_p_weight=log_p_weight, dec_weight=dec_weight)

        # Skip batch se loss è NaN
        if any(np.isnan(v) for v in loss_dict.values()):
          self.optimizer.zero_grad()  # annulla i gradienti
          return {k: 0.0 for k in loss_dict}
        
        #nn.utils.clip_grad_value_([p for p in self.get_all_parameters() if p.requires_grad], 1.0)
        nn.utils.clip_grad_norm_([p for p in self.get_all_parameters() if p.requires_grad], max_norm=1.0)
        self.optimizer.step()
        return loss_dict
        
    def get_all_parameters(self):
        params = list(self.proj_v.parameters()) + list(self.proj_t.parameters()) + \
                 list(self.deproj_v.parameters()) + list(self.deproj_t.parameters()) + \
                 list(self.flow_v.parameters()) + list(self.flow_t.parameters()) + \
                 list(self.rec_v.parameters()) + list(self.rec_t.parameters()) + \
                 list(self.cat_v.parameters()) + list(self.cat_t.parameters())
        return params
    
    def impute_textual(self, visual_feature): # Imputa textual feature da visual feature
        if visual_feature.dim() == 1:
            visual_feature = visual_feature.unsqueeze(0)
        visual_feature = visual_feature.to(self.device)
        
        self.proj_v.eval()
        self.proj_t.eval()
        self.deproj_t.eval()
        self.flow_v.eval()
        self.flow_t.eval()
        self.rec_t.eval()
        with torch.no_grad():
            # Project visual
            x_v = visual_feature.unsqueeze(-1)  # [1, visual_dim, 1]
            x_v = x_v.squeeze(1)
            proj_x_v = self.proj_v(x_v)  # [1, embd_size, 1]
            # Flow forward visual
            proj_x_v_4d = proj_x_v.unsqueeze(-1)  # [1, embd_size, 1, 1]
            _, _, z_v = self.flow_v(proj_x_v_4d)
            # Reverse flow per textual (cross-modal transfer)
            proj_x_t = self.flow_t.reverse(z_v, reconstruct=True).squeeze(-1)  # [1, embd_size, 1]
            # Refine con RCAN
            proj_x_t = self.rec_t(proj_x_t)  # [1, embd_size, 1]
            # Output: Deproject a dimensione originale
            textual_imputed_full = self.deproj_t(proj_x_t)  # [1, textual_dim, 1]
            textual_imputed = textual_imputed_full.squeeze(-1).squeeze(0)  # [textual_dim]
        return textual_imputed
    
    def impute_visual(self, textual_feature): #Imputa visual feature da textual feature
        if textual_feature.dim() == 1:
            textual_feature = textual_feature.unsqueeze(0)
        textual_feature = textual_feature.to(self.device)
        
        self.proj_v.eval()
        self.proj_t.eval()
        self.deproj_v.eval()
        self.flow_v.eval()
        self.flow_t.eval()
        self.rec_v.eval()
        with torch.no_grad():
            # Project textual
            x_t = textual_feature.unsqueeze(-1)  # [1, textual_dim, 1]
            x_t = x_t.squeeze(1)
            proj_x_t = self.proj_t(x_t)  # [1, embd_size, 1]
            # Flow forward textual
            proj_x_t_4d = proj_x_t.unsqueeze(-1)  # [1, embd_size, 1, 1]
            _, _, z_t = self.flow_t(proj_x_t_4d)
            # Reverse flow per visual (cross-modal transfer)
            proj_x_v = self.flow_v.reverse(z_t, reconstruct=True).squeeze(-1)  # [1, embd_size, 1]
            # Refine con RCAN
            proj_x_v = self.rec_v(proj_x_v)  # [1, embd_size, 1]
            # Output: Deproject a dimensione originale
            visual_imputed_full = self.deproj_v(proj_x_v)  # [1, visual_dim, 1]
            visual_imputed = visual_imputed_full.squeeze(-1).squeeze(0)  # [visual_dim]
        return visual_imputed
    
    def train(self): #Set networks to training mode
        self.proj_v.train()
        self.proj_t.train()
        self.deproj_v.train() 
        self.deproj_t.train()
        self.flow_v.train()
        self.flow_t.train()
        self.rec_v.train()
        self.rec_t.train()
        self.cat_v.train()
        self.cat_t.train()
    
    def eval(self): #Set networks to evaluation mode
        self.proj_v.eval()
        self.proj_t.eval()
        self.deproj_v.eval() 
        self.deproj_t.eval()
        self.flow_v.eval()
        self.flow_t.eval()
        self.rec_v.eval()
        self.rec_t.eval()
        self.cat_v.eval()
        self.cat_t.eval()