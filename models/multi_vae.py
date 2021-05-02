import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MultiVAE(nn.Module):
   def __init__(self, p_dims, q_dims=None, dropout_p=0.5):
      super().__init__()
      self.p_dims = p_dims
      
      if q_dims:
         assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
         assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
         self.q_dims = self.q_dims
      else:
         self.q_dims = p_dims[::-1]

      # Encoder Layers
      # Get Mean and Variance, Last Dimension Incrased.
      temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
      self.q_layers = nn.ModuleList(
         [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

      # Decoder Layers
      self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(
         self.p_dims[:-1], self.p_dims[1:])])

      self.drop = nn.Dropout(dropout_p)

      # Initilization
      self.init_weights(self.q_layers)
      self.init_weights(self.p_layers)

   def reparameterize(self, mu, logvar):
      if self.training:
         std = torch.exp(0.5 * logvar)
         eps = torch.randn_like(std)
         return eps*std + mu
      else:
         return mu

   def encoder(self, x):
      h = F.normalize(x, p=2, dim=1)  # L2-Normalize
      h = self.drop(h)
      for i, layer in enumerate(self.q_layers):
         h = layer(h)
         if i != len(self.q_layers) - 1:
               h = torch.tanh(h)
         else:
               mu = h[:, :self.q_dims[-1]]
               logvar = h[:, self.q_dims[-1]:]
      return mu, logvar

   def decoder(self, z):
      h = z
      for i, layer in enumerate(self.p_layers):
         h = layer(h)
         if i != len(self.p_layers) - 1:
               h = torch.tanh(h)
      return h

   def forward(self, x):
      mu, logvar = self.encoder(x)
      z = self.reparameterize(mu, logvar)
      recon_x = self.decoder(z)
      return recon_x, mu, logvar

   def init_weights(self, m):
      for layer in m:
         if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias, 0, 0.001)
            
   def loss_function(self, recon_x, x, mu, logvar, anneal=1):
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, dim=1) * x, -1))
      return neg_ll + anneal * KLD
