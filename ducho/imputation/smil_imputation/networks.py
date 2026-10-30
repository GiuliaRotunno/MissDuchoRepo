# SMIL networks adapted for feature vectors from https://github.com/deep-real/SMIL
import torch
import torch.nn as nn
from collections import OrderedDict

# Encoder that generates perturbation parameters for each layer during meta-training.
# Changes from the original: removed conv1, conv2; kept only fc layers for feature vectors; adjusted sizes for visual (2048) and textual (768)
class InferNet(nn.Module):    

    def __init__(self, visual_dim=2048, textual_dim=768):
        super(InferNet, self).__init__() 
        
        # Concatenated size of the two modalities
        concat_dim = visual_dim + textual_dim  # 2048 + 768 = 2816
        # Encoders to generate perturbations at different levels
        self.fc_concat = nn.Linear(concat_dim, concat_dim)   # fc_concat: perturbation on the concatenated feature
        self.fc1 = nn.Linear(1024, 1024)  # fc1: perturbation on the first hidden layer (after fusion)       
        self.fc2 = nn.Linear(512, 512)    # fc2: perturbation on the second secondo hidden layer
        self.fc3 = nn.Linear(256, 256)    # fc3: perturbation on the third hidden layer
    
    def forward(self, x_dict, meta_train=True):
        outputs = OrderedDict()   # outputs: dictionary with perturbation parameters for each layer
        
        if meta_train:  # meta_train: if True, add Gaussian noise (training); if False, use only mu (testing/imputation).
            # During meta-training: mu + Gaussian noise
            if 'fc_concat' in x_dict.keys():                      # x_dict: dictionary with features from different layers
                mu = self.fc_concat(x_dict['fc_concat'])
                outputs['fc_concat'] = torch.randn_like(mu) + mu
            if 'fc1' in x_dict.keys():
                mu = self.fc1(x_dict['fc1'])
                outputs['fc1'] = torch.randn_like(mu) + mu
            if 'fc2' in x_dict.keys():
                mu = self.fc2(x_dict['fc2'])
                outputs['fc2'] = torch.randn_like(mu) + mu
            if 'fc3' in x_dict.keys():
                mu = self.fc3(x_dict['fc3'])
                outputs['fc3'] = torch.randn_like(mu) + mu
        else:
            # During testing/imputation: mu only (no noise)
            if 'fc_concat' in x_dict.keys():
                mu = self.fc_concat(x_dict['fc_concat'])
                outputs['fc_concat'] = mu
            if 'fc1' in x_dict.keys():
                mu = self.fc1(x_dict['fc1'])
                outputs['fc1'] = mu
            if 'fc2' in x_dict.keys():
                mu = self.fc2(x_dict['fc2'])
                outputs['fc2'] = mu
            if 'fc3' in x_dict.keys():
                mu = self.fc3(x_dict['fc3'])
                outputs['fc3'] = mu

        return outputs

# Modality Fuser: merges visual and textual modalities and applies Bayesian perturbations
# Two modes of operation: mode='one' (one mode available + average of the other + perturbations), mode='two' (both modes available, no perturbations)
class ModalityFuser(nn.Module): 
   
    def __init__(self, visual_dim=2048, textual_dim=768, num_classes=None):
        super(ModalityFuser, self).__init__()
        self.visual_dim = visual_dim
        self.textual_dim = textual_dim
        concat_dim = visual_dim + textual_dim  # 2816
        
        # Blending layers (as in the original but with adapted dimensions)
        self.fc1 = nn.Linear(concat_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        # Decoder heads to reconstruct in the original space
        self.decoder_visual = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, visual_dim)  # 1024 → 2048
        )
        self.decoder_textual = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, textual_dim)
        )

        # Output layer (optional, for classification tasks)
        if num_classes is not None:
            self.fc_out = nn.Linear(256, num_classes)
        else:
            self.fc_out = None

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
    
    # Forward pass with two modes
    def forward(self, visual_feature=None, textual_feature=None, encoder=None, textual_mean=None, visual_mean=None, 
                noise_layers=['fc_concat', 'fc1', 'fc2', 'fc3'], meta_train=True, mode='one', missing_modality='textual'):
        
        if mode == 'one':  # One modality + average of the other + perturbations
            assert encoder is not None, "encoder required for mode='one'"   # encoder: InferNet to generate perturbations
            if missing_modality == 'textual': # Visual available, textual missing
                assert textual_mean is not None, "textual_mean required when missing_modality='textual'"
                assert visual_feature is not None, "visual_feature required when missing_modality='textual'"
                textual_feature = textual_mean.expand(visual_feature.shape[0], textual_mean.shape[0]) # Expands the textual average for the entire batch
                visual_feature = visual_feature.squeeze(1)
                x = torch.cat([visual_feature, textual_feature], dim=1)  # Concatenate visual + textual
            elif missing_modality == 'visual':  # Textual available, visual missing
                assert visual_mean is not None, "visual_mean required when missing_modality='visual'"
                assert textual_feature is not None, "textual_feature required when missing_modality='visual'"
                visual_feature = visual_mean.expand(textual_feature.shape[0], visual_mean.shape[0])   # Expands the visual average for the entire batch
                textual_feature = textual_feature.squeeze(1)
                x = torch.cat([visual_feature, textual_feature], dim=1)  # Concatenate visual + textual
            else:
                raise ValueError(f"missing_modality must be 'visual' or 'textual', got {missing_modality}")
    
            # Apply perturbation to the concatenated level
            if 'fc_concat' in noise_layers:   # list of layers where to apply perturbations
                fc_concat_dict = {'fc_concat': x}
                perturbation = encoder(fc_concat_dict, meta_train=meta_train)['fc_concat']  # meta_train: if True uses noise, if False not
                x = self.softplus(perturbation) * x
            f_concat = x
            
            # First hidden layer
            x = self.fc1(x)
            if 'fc1' in noise_layers:
                fc1_dict = {'fc1': x}
                perturbation = encoder(fc1_dict, meta_train=meta_train)['fc1']
                x = self.softplus(perturbation) * x
            f1 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn1(x)

            # Second hidden layer
            x = self.fc2(x)
            if 'fc2' in noise_layers:
                fc2_dict = {'fc2': x}
                perturbation = encoder(fc2_dict, meta_train=meta_train)['fc2']
                x = self.softplus(perturbation) * x
            f2 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn2(x)

            # Third hidden layer
            x = self.fc3(x)
            if 'fc3' in noise_layers:
                fc3_dict = {'fc3': x}
                perturbation = encoder(fc3_dict, meta_train=meta_train)['fc3']
                x = self.softplus(perturbation) * x
            f3 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn3(x)

            # Output layer (optional)
            if self.fc_out is not None:
                x = self.fc_out(x)
            visual_reconstructed = self.decoder_visual(x)     # [batch, visual_dim]
            textual_reconstructed = self.decoder_textual(x)   # [batch, textual_dim]
            return x, [f_concat, f1, f2, f3], visual_reconstructed, textual_reconstructed       
        
        elif mode == 'two':   # Both modalities available, no perturbations
            assert textual_feature is not None, "textual_feature required for mode='two'"  # 2,1,768
            assert visual_feature is not None, "visual_feature required for mode='two'"    # 2,1,2048
            # Concatenate visual + textual
            x = torch.cat([visual_feature, textual_feature], dim=1)
            f_concat = x
            
            # First hidden layer
            x = self.fc1(x)
            f1 = x
            x = self.relu(x)
            x = self.dropout(x)
            #x = x.squeeze(1) 
            x = self.bn1(x)
            
            # Second hidden layer
            x = self.fc2(x)
            f2 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn2(x)

            # Third hidden layer
            x = self.fc3(x)
            f3 = x
            x = self.relu(x)
            x = self.dropout(x)
            x = self.bn3(x)

            # Output layer (optional)
            if self.fc_out is not None:
                x = self.fc_out(x)
            visual_reconstructed = self.decoder_visual(x)
            textual_reconstructed = self.decoder_textual(x)
            return x, [f_concat, f1, f2, f3], visual_reconstructed, textual_reconstructed

        else:
            raise ValueError(f"mode should be 'one' or 'two', got {mode}")

# Initialize network weights
def weights_init(m):  

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def create_infernet(visual_dim=2048, textual_dim=768, gpu_ids=[]):

    net = InferNet(visual_dim, textual_dim)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)

    return net

def create_fuser(visual_dim=2048, textual_dim=768, num_classes=None, gpu_ids=[]):

    net = ModalityFuser(visual_dim, textual_dim, num_classes)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)
    
    return net
