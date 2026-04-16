import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class AutoEncoderVisual2Textual(nn.Module):
    def __init__(self):
        super(AutoEncoderVisual2Textual, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoderTextual2Visual(nn.Module):
    def __init__(self):
        super(AutoEncoderTextual2Visual, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 1024),   
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded   
    
class GraphEncoderVisual2Textual(nn.Module):
    def __init__(self):
        super(GraphEncoderVisual2Textual, self).__init__()
        self.conv1 = GCNConv(2048, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        encoded = self.conv2(encoded, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
     
class GraphEncoderTextual2Visual(nn.Module):
    def __init__(self):
        super(GraphEncoderTextual2Visual, self).__init__()
        self.conv1 = GCNConv(768, 1024)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def message_passing(self, x, edge_index):
        encoded = self.conv1(x, edge_index)
        encoded = nn.functional.leaky_relu(encoded)
        return encoded

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

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

class VariationalGraphEncoderVisual2Textual(nn.Module):
    def __init__(self):
        super(VariationalGraphEncoderVisual2Textual, self).__init__()       
        self.conv1 = GCNConv(2048, 1024)  # Encoder condiviso
        self.conv_mu = GCNConv(1024, 512)  # due teste (mu, logvar)
        self.conv_logvar = GCNConv(1024, 512)
        self.decoder = nn.Sequential(  # Decoder invariato
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 768),
            nn.Sigmoid()
        )

    def encode(self, x, edge_index):
        h = F.leaky_relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar
    
class VariationalGraphEncoderTextual2Visual(nn.Module):
    def __init__(self):
        super(VariationalGraphEncoderTextual2Visual, self).__init__()
        self.conv1 = GCNConv(768, 1024)
        self.conv_mu = GCNConv(1024, 1024)
        self.conv_logvar = GCNConv(1024, 1024)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def encode(self, x, edge_index):
        h = F.leaky_relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

