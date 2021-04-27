import torch 
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl

class MultiVAE(pl.LightningModule):
   def __init__(self, p_dims, q_dims=None, dropout_p=0.5, **kwargs):
      super().__init__()
      self.save_hyperparameters()
      
      if self.hparams.q_dims:
         assert self.hparams.q_dims[0] == self.hparmas.p_dims[-1], "In and Out dimensions must equal to each other"
         assert self.hparmas.q_dims[-1] == self.hparmas.p_dims[0], "Latent dimension for p- and q- network mismatches."
         self.q_dims = self.hparmas.q_dims
      else:
         self.q_dims = self.hparmas.p_dims[::-1]

      # Encoder Layers
      temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2] # Get Mean and Variance, Last Dimension Incrased.
      self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
      
      # Decoder Layers
      self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.hparams.p_dims[:-1], self.hparams.p_dims[1:])])
      
      self.drop = nn.Dropout(self.hparams.dropout_p)
      
      # Initilization
      self.init_weights.apply(self.q_layers)
      self.init_weights.apply(self.p_layers)
      

   def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std
   
   def encoder(self, x):
      h = F.normalize(x, p=2, dim=1)  # L2-Normalize
      h = self.drop(h)
      for i, layer in enumerate(self.q_layers):
         h = layer(h)
         if i != len(self.q_layers) - 1:
            h = F.tanh(h)
         else:
            mu = h[:, :self.q_dims[-1]]
            logvar = h[:, self.q_dims[-1]:]
      return mu, logvar
   
   def decoder(self, z):
      for i, layer in enumerate(self.p_layers):
         h = layer(z)
         if i != len(self.p_layers) - 1:
            h = F.tanh(h)
      return h
   
   def forward(self, x):
      mu, logvar = self.encoder(x)
      z = self.reparameterize(mu, logvar)
      recon_x = self.decoder(z)
      return recon_x, mu, logvar
   
   def init_weights(self, m):
      if type(m) == nn.Linear:
         torch.nn.init.xavier_normal_(m.weight)
         torch.nn.init.normal_(m.bias, 0, 0.001)
      

   def loss_function(self, recon_x, x, mu, logvar):
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, dim=1) * x, -1))
      return BCE + self.hparams.anneal * KLD
