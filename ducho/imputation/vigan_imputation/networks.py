# VIGAN networks adapted for feature vectors from https://github.com/chaoshangcs/VIGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multimodal AutoEncoder
# Changes: Removed view(-1, 784) because we already work with vectors, parameterized dimensions (input_A, input_B) instead of hardcoded, same layer scheme as the original
class AutoEncoder(nn.Module): 

    def __init__(self, input_A, input_B, hidden_dim=400, shared_dim=200, dropout=0.5):
        super(AutoEncoder, self).__init__()
        self.layer1_1 = nn.Linear(input_A, hidden_dim, bias=False)  # Encoder for modality A (visual)
        self.layer1_2 = nn.Linear(input_B, hidden_dim, bias=False)  # Encoder for modality B (textual)
        # Shared layers
        self.layer2 = nn.Linear(hidden_dim * 2, 500)
        self.layer3 = nn.Linear(500, shared_dim)
        self.layer4 = nn.Linear(shared_dim, hidden_dim)
        # Decoder branches
        self.layer5_1 = nn.Linear(hidden_dim, 500)
        self.layer5_2 = nn.Linear(hidden_dim, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(500, input_B)
        self.drop = dropout

    # Forward pass
    def forward(self, x1, x2): 
        # Encode
        x1 = F.dropout(F.relu(self.layer1_1(x1)), self.drop, training=self.training)  #x1: visual features [batch, input_A]
        x2 = F.dropout(F.relu(self.layer1_2(x2)), self.drop, training=self.training)  #x2: textual features [batch, input_B]
        # Shared representation
        x = F.dropout(F.relu(self.layer2(torch.cat((x1, x2), 1))), self.drop, training=self.training)
        x = F.dropout(F.relu(self.layer3(x)), self.drop, training=self.training)
        x = F.dropout(F.relu(self.layer4(x)), self.drop, training=self.training)
        # Decode to both modalities
        out1 = F.relu(self.layer5_1(x))
        out1 = torch.sigmoid(self.layer6_1(out1))     # out1: reconstructed visual [batch, input_A]
        out2 = F.relu(self.layer5_2(x))
        out2 = torch.sigmoid(self.layer6_2(out2))     # out2: reconstructed textual [batch, input_B]
        return out1, out2   

# Inizialize network weights
def weights_init(m):  
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Create AutoEncoder 
def define_AE(input_A, input_B, gpu_ids=[]):

    NetAE = AutoEncoder(input_A, input_B)  # input_A: modality A dimension (visual) - input_B: modality B dimension (textual)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        NetAE = NetAE.cuda(gpu_ids[0])
    NetAE.apply(weights_init)

    return NetAE

# Create Generator
def define_G(input_dim, output_dim, gpu_ids=[]): 

    netG = Generator(input_dim, output_dim)           
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        netG = netG.cuda(gpu_ids[0])  
    netG.apply(weights_init)

    return netG

# Create Discriminator   
def define_D(input_dim, gpu_ids=[]): 

    netD = Discriminator(input_dim)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        netD = netD.cuda(gpu_ids[0])
    netD.apply(weights_init)

    return netD

# GAN Loss, supports LSGAN (MSE) or vanilla GAN (BCE)
class GANLoss(nn.Module):  

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.size() != input.size():
                self.real_label_var = torch.full(input.size(), self.real_label, device=input.device, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            if self.fake_label_var is None or self.fake_label_var.size() != input.size():
                self.fake_label_var = torch.full(input.size(), self.fake_label, device=input.device, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Generator for feature vectors: map from one modality to another (e.g. visual -> textual); the original uses ResNet/UNet for images, here we use MLP for vectors.
class Generator(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256], dropout=0.3):
        super(Generator, self).__init__()
        layers = []
        prev_dim = input_dim
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        # Output layer with Tanh
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Discriminator for feature vectors: distinguishes between real and generated features; the original uses NLayerDiscriminator (PatchGAN) for images, here we use MLP for vectors.
class Discriminator(nn.Module):  
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3):
        super(Discriminator, self).__init__()
        layers = []
        prev_dim = input_dim
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        # Output layer: real/fake probability
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
