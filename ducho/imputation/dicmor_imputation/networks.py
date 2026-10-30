#DiCMoR networks adapted for feature vectors form https://github.com/mdswyz/DiCMoR
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la
logabs = lambda x: torch.log(torch.abs(x))

# RCAN - Residual Channel Attention Network 

# Channel Attention Layer 
class CALayer(nn.Module):

    def __init__(self, num_channels, reduction=16): 
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block
class RCAB(nn.Module): 

    def __init__(self, num_channels, reduction, res_scale):
        super().__init__()
        body = [
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, 3, 1, 1),
        ]
        body.append(CALayer(num_channels, reduction))
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

#Group of RCAB blocks
class RCANGroup(nn.Module): 

    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        super().__init__()
        body = []
        for _ in range(num_blocks):
            body += [RCAB(num_channels, reduction, res_scale)]
        body += [nn.Conv1d(num_channels, num_channels, 3, 1, 1)]
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# Glow Components - Normalizing Flow 

# Activation Normalization
class ActNorm(nn.Module): 

    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

#Invertible 1x1 Convolution with LU decomposition
class InvConv2dLU(nn.Module): 

    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
    
# 2D Convolution
class InvConv2d(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = (height * width * torch.slogdet(self.weight.squeeze().double())[1].float())
        return out, logdet

    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

# Zero-initialized 2D Convolution
class ZeroConv2d(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out

# Affine Coupling Layer
class AffineCoupling(nn.Module): 

    def __init__(self, in_channel, filter_size=64, affine=True):
        super().__init__()
        self.affine = affine
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            log_s = torch.clamp(log_s, -5, 5)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        
        return torch.cat([out_a, in_b], 1)

# Flow step
class Flow(nn.Module):

    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

# Gaussian log probability
def gaussian_log_p(x, mean, log_sd): 
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

# Sample from Gaussian
def gaussian_sample(eps, mean, log_sd): 
    return mean + torch.exp(log_sd) * eps

# Glow Block 
class Block(nn.Module): 

    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(in_channel, affine=affine, conv_lu=conv_lu))     
        self.prior = ZeroConv2d(in_channel, in_channel * 2) # Simplified prior: not conditioned on the class

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        out = input
        logdet = 0
        
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det
        
        # Not conditioned prior
        zero = torch.zeros_like(out)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        log_p = gaussian_log_p(out, mean, log_sd)
        log_p = log_p.view(b_size, -1).sum(1)
        z_new = out
        
        return out, logdet, log_p, z_new

    # Reverse flow for reconstruction
    def reverse(self, output, eps=None, reconstruct=False): 
        input = output
        
        if reconstruct:
            input = output # Use output directly for reconstruction
        else:
            # Sample from prior
            zero = torch.zeros_like(input)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = z
        
        # Reverse through the flows
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        
        return input
  
# Glow: Generative Flow  (Simplified prior: not conditioned on the class)
class Glow(nn.Module):

    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(Block(in_channel, n_flow, affine=affine, conv_lu=conv_lu))

    def forward(self, input): #Forward: data → latent
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        
        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False): #Reverse: latent → data
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        
        return input

# Helper Functions

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Create Glow network
def create_glow(in_channel, n_flow=32, n_block=1, gpu_ids=[]): 
    net = Glow(in_channel, n_flow, n_block, affine=True, conv_lu=False)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    return net

# Create RCAN refinement network
def create_rcan_refiner(in_channel, num_blocks=20, reduction=16, gpu_ids=[]): 
    net = nn.Sequential(
        nn.Conv1d(in_channel, in_channel*2, 1),
        RCANGroup(num_channels=in_channel*2, num_blocks=num_blocks, reduction=reduction),
        nn.Conv1d(in_channel*2, in_channel, 1)
    )
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        net = net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net
