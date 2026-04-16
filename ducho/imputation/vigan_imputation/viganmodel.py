# VIGAN Model adattato per vettori di feature, estraendo il core dell'algoritmo e rimuovedo le dipndenze https://github.com/chaoshangcs/VIGAN/blob/master/models/VIGAN_model.py
import torch
import torch.nn as nn
import itertools
from ducho.imputation.vigan_imputation.networks import *

class VIGANModel: #VIGAN Model per imputazione di modalità mancanti, estratto dal codice originale VIGAN, adattato per feature vectors.
    #Training in 3 fasi (come nell'originale):  1. Pre-train CycleGAN (G_A, G_B)   2. Pre-train Autoencoder (AE)  3. Joint optimization (AE + G_A + G_B)
    def __init__(self, input_A_dim, input_B_dim, gpu_ids=[0], use_lsgan=True):
        self.input_A_dim = input_A_dim  #input_A_dim: dimensione modalità A (visual, es. 2048) 
        self.input_B_dim = input_B_dim  #input_B_dim: dimensione modalità B (textual, es. 768)   
        self.gpu_ids = gpu_ids          #gpu_ids: lista GPU IDs (es. [0] per cuda:0) 
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 and torch.cuda.is_available() else 'cpu') #use_lsgan: True per LSGAN (MSE), False per vanilla GAN (BCE)
        
        # Crea le reti     
        self.netG_A = define_G(input_A_dim, input_B_dim, gpu_ids) #Generator G_A: A->B
        self.netG_B = define_G(input_B_dim, input_A_dim, gpu_ids) #Generator G_B: B->A
        self.AE = define_AE(input_A_dim, input_B_dim, gpu_ids) # Autoencoder
        self.netD_A = define_D(input_B_dim, gpu_ids)  #Discriminator per modalità B
        self.netD_B = define_D(input_A_dim, gpu_ids)  #Discriminator per modalità A
        
        # Loss functions e sposta tutto sul device
        self.criterionGAN = GANLoss(use_lsgan=use_lsgan)
        self.criterionGAN = self.criterionGAN.to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionCycle = self.criterionCycle.to(self.device)
        self.criterionIdt = nn.L1Loss()
        self.criterionIdt = self.criterionIdt.to(self.device)
        self.criterionAE = nn.MSELoss()
        self.criterionAE = self.criterionAE.to(self.device)
        
        # Variabili per memorizzare dati e loss
        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.rec_A = None
        self.rec_B = None
        self.AEfakeA = None
        self.AEfakeB = None
    
    def setup_optimizers(self, lr=0.0002, beta1=0.5):
        # Optimizer per CycleGAN pre-training
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=lr, betas=(beta1, 0.999))
        # Optimizer per AE pre-training
        self.optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=lr, betas=(beta1, 0.999))
        # Optimizer per joint optimization
        self.optimizer_D_A_AE = torch.optim.Adam(self.netD_A.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_B_AE = torch.optim.Adam(self.netD_B.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_AE_GA_GB = torch.optim.Adam(itertools.chain(self.AE.parameters(), self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
    
    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)  #real_A: batch di features modalità A [batch, input_A_dim] 
        self.real_B = real_B.to(self.device)  #real_B: batch di features modalità B [batch, input_B_dim]

    # Backward functions (estratte dall'originale)   
    def backward_D_basic(self, netD, real, fake):  #Basic discriminator backward 
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self): #Discriminator A backward 
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
    
    def backward_D_B(self): #Discriminator B backward
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
    
    def backward_G(self, lambda_A=10.0, lambda_B=10.0, lambda_idt=0.5):  #Generator backward per CycleGAN
        # Identity loss
        if lambda_idt > 0:  #lambda_idt: peso identity loss
            # G_A should be identity if real_B is fed
            self.idt_A = self.netG_A(self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt  #lambda_B: peso cycle loss B
            # G_B should be identity if real_A is fed
            self.idt_B = self.netG_B(self.real_B) 
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt  #lambda_A: peso cycle loss A 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # GAN loss
        self.fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        self.fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        self.rec_A = self.netG_B(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def backward_AE_pretrain(self):  #Autoencoder pre-training backward
        AErealA, AErealB = self.AE(self.real_A, self.real_B)
        self.loss_AE_pre = self.criterionAE(AErealA, self.real_A) + self.criterionAE(AErealB, self.real_B)
        self.loss_AE_pre.backward()
    
    def backward_AE(self): #Autoencoder backward durante joint training
        # Fake data
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)
        # Autoencoder loss: fakeA
        AEfakeA, AErealB = self.AE(self.fake_A, self.real_B)
        self.loss_AE_fA_rB = (self.criterionAE(AEfakeA, self.real_A) + self.criterionAE(AErealB, self.real_B))
        # Autoencoder loss: fakeB
        AErealA, AEfakeB = self.AE(self.real_A, self.fake_B)
        self.loss_AE_rA_fB = (self.criterionAE(AErealA, self.real_A) + self.criterionAE(AEfakeB, self.real_B))
        # Combined loss
        self.loss_AE = (self.loss_AE_fA_rB + self.loss_AE_rA_fB) * 0.5
        self.loss_AE.backward()
    
    def backward_D_A_AE(self): #Discriminator A per joint training 
        self.loss_D_A_AE = self.backward_D_basic(self.netD_A, self.real_B, self.AEfakeB)
    
    def backward_D_B_AE(self): #Discriminator B per joint training
        self.loss_D_B_AE = self.backward_D_basic(self.netD_B, self.real_A, self.AEfakeA)
    
    def backward_AE_GA_GB(self, lambda_C=2.0, lambda_D=10.0): #Joint backward per AE+GA+GB   #1.0  1.0
        # Fake data
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)
        # Forward cycle loss
        self.rec_A = self.netG_B(self.fake_B)
        self.loss_cycle_A_AE = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B_AE = self.criterionCycle(self.rec_B, self.real_B)
        # Autoencoder loss: fakeA
        self.AEfakeA, AErealB = self.AE(self.fake_A, self.real_B)
        self.loss_AE_fA_rB = (self.criterionAE(self.AEfakeA, self.real_A) + self.criterionAE(AErealB, self.real_B))
        # Autoencoder loss: fakeB
        AErealA, self.AEfakeB = self.AE(self.real_A, self.fake_B)
        self.loss_AE_rA_fB = (self.criterionAE(AErealA, self.real_A) +  self.criterionAE(self.AEfakeB, self.real_B))
        self.loss_AE = (self.loss_AE_fA_rB + self.loss_AE_rA_fB)
        # D loss
        pred_fake = self.netD_A(self.AEfakeB)
        self.loss_AE_GA = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(self.AEfakeA)
        self.loss_AE_GB = self.criterionGAN(pred_fake, True)
        # Combined loss
        self.loss_AE_GA_GB = (lambda_C * (self.loss_AE_GA + self.loss_AE_GB) + lambda_D * self.loss_AE + 1.0 * (self.loss_cycle_A_AE + self.loss_cycle_B_AE))  #lambda_C: peso per GAN loss  #lambda_D: peso per AE reconstruction loss
        self.loss_AE_GA_GB.backward()

    # Optimize functions (estratte dall'originale)    
    def optimize_parameters_pretrain_cycleGAN(self, lambda_A=10.0, lambda_B=10.0, lambda_idt=0.5): #Pre-train CycleGAN [FASE 1 del training]
        #G_A e G_B
        self.optimizer_G.zero_grad()
        self.backward_G(lambda_A, lambda_B, lambda_idt)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
    
    def optimize_parameters_pretrain_AE(self): #Pre-train Autoencoder [FASE 2 del training]
        self.optimizer_AE.zero_grad()
        self.backward_AE_pretrain()
        self.optimizer_AE.step()
    
    def optimize_parameters(self, lambda_C=1.0, lambda_D=1.0, n_iter_G=20, n_iter_D=20): #Joint optimization [FASE 3 del training]    #2 1
        # AE+G_A+G_B
        for i in range(n_iter_G): #n_iter_G: numero iterazioni generator
            self.optimizer_AE_GA_GB.zero_grad()
            self.backward_AE_GA_GB(lambda_C, lambda_D) #lambda_C: peso GAN loss  #lambda_D: peso AE loss
            self.optimizer_AE_GA_GB.step()
        # D_A and D_B
        for i in range(n_iter_D): # n_iter_D: numero iterazioni discriminator
            self.optimizer_D_A_AE.zero_grad()
            self.backward_D_A_AE()
            self.optimizer_D_A_AE.step()
            self.optimizer_D_B_AE.zero_grad()
            self.backward_D_B_AE()
            self.optimizer_D_B_AE.step()
    
    # Utility functions
    def set_requires_grad(self, nets, requires_grad=False): #Set requires_grad for networks
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def eval(self): #Set networks to evaluation mode
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        self.AE.eval()
    
    def train(self): #Set networks to training mode
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()
        self.AE.train()