import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDAE(nn.Module):
   def __init__(self, p_dims, q_dims=None, dropout_p=0.5):
      super().__init__()
      self.p_dims = p_dims

      if q_dims:
         assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
         assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
         self.q_dims = self.q_dims
      else:
         self.q_dims = p_dims[::-1]

      # Layers
      self.dims = self.q_dims + self.p_dims[1:]
      self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                   d_in, d_out in zip(self.dims[:-1], self.dims[1:])])

      self.drop = nn.Dropout(dropout_p)

      # Initilization
      self.init_weights(self.layers)

   def forward(self, x):
      h = F.normalize(x, p=2, dim=1)
      h = self.drop(h)
      for i, layer in enumerate(self.layers):
         h = layer(h)
         if i != len(self.layers) - 1:
            h = torch.tanh(h)
      return h

   def init_weights(self, m):
      for layer in m:
         if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias, 0, 0.001)
            
   def loss_function(self, recon_x, x):
      neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, dim=1) * x, -1))
      return neg_ll
