#MMIN networks adapted for feature vectors from https://github.com/AIM3-RUC/MMIN
import torch
import torch.nn as nn
import copy
 
# Residual Autoencoder: use multiple encoder-decoder blocks with residual connections
class ResidualAE(nn.Module):   
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn        # use_bn: whether to use batch normalization
        self.dropout = dropout      # dropout: dropout rate
        self.n_blocks = n_blocks    # n_blocks: number of encoder-decoder blocks
        self.input_dim = input_dim  # input_dim: input dimension (visual_dim + textual_dim)
        # Final transition layer
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):     # Create n_blocks encoder-decoder pairs
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))

    # Create encoder   
    def get_encoder(self, layers):  

        all_layers = []
        input_dim = self.input_dim
        for i in range(len(layers)):  #layers: list of hidden layers dimensions
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # Remove activation of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]

        return nn.Sequential(*all_layers)

    # Create decoder 
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))

        return nn.Sequential(*all_layers)

    # Forward pass
    def forward(self, x):  
        x_in = x                         # x: input tensor [batch, input_dim]
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):   # Go through all blocks with residual connections
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            # Residual connection
            x_in = x_in + x_out
            # Encode-decode
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        # Concatenate all latent codes
        latents = torch.cat(latents, dim=-1)      # latents: all latent codes concatenated [batch, sum(layers[-1])*n_blocks]
        # Final transition
        recon = self.transition(x_in + x_out)     # recon: final reconstruction[batch, input_dim]
        return recon, latents 

# Modality Encoders - Single modality encoder: adapted for feature vectors, MLP reducing modality size to embd_size.   
class ModalityEncoder(nn.Module):  

    def __init__(self, input_dim, embd_size, hidden_dim=None, dropout=0.5):
        super(ModalityEncoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + embd_size) // 2   # hidden_dim: hidden layer dimension (default: average between input and output)
        self.encoder = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim),   #input_dim: input dimension (2048 for visual, 768 for textual)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embd_size),   #embd_size: output embedding dimension (128)
            nn.ReLU()
        )

    # Forward pass 
    def forward(self, x):        # x: input tensor [batch, input_dim]
        return self.encoder(x)   # embd: embedding [batch, embd_size]

# Decoder that projects from embed size to original size
class ModalityDecoder(nn.Module):  

    def __init__(self, embd_size, output_dim, hidden_dim=None, dropout=0.5):
        super(ModalityDecoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = (embd_size + output_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(embd_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.decoder(x)


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
 
def create_residual_ae(input_dim, layers=[128, 64, 32], n_blocks=3, dropout=0.5, use_bn=False, gpu_ids=[]):

    net = ResidualAE(layers, n_blocks, input_dim, dropout, use_bn)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)

    return net

def create_modality_encoder(input_dim, embd_size, hidden_dim=None, dropout=0.5, gpu_ids=[]):

    net = ModalityEncoder(input_dim, embd_size, hidden_dim, dropout)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)

    return net

def create_modality_decoder(embd_size, output_dim, hidden_dim=None, dropout=0.5, gpu_ids=[]):

    net = ModalityDecoder(embd_size, output_dim, hidden_dim, dropout)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)
    
    return net 