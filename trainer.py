import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl

from models import MultiVAE
from models import MultiDAE
from metric import Recall_at_k_batch, NDCG_binary_at_k_batch

logger = logging.getLogger(__name__)
writer = SummaryWriter()

MODEL_LIST = {
    "multi-vae": MultiVAE,
    "multi-dae": MultiDAE
}

class Trainer(object):
   def __init__(self, args, p_dims):
      self.args = args
      self.p_dims = p_dims
      
      # Model Select
      if args.model_name not in MODEL_LIST.keys():
         raise ValueError(
             "Please choose the model selected in the list: " + ", ".join(MODEL_LIST.keys()))
      self.model = MODEL_LIST[args.model_name](self.p_dims)
      
      # Optimizer 
      self.optimizer = torch.optim.Adam(
          self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      
      # CPU or GPU
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      self.model.to(self.device)
      
      self.epochs = 0
      self.update_count = 0.0

   def train(self, train_dataloader):
      # Turn on training mode
      train_loss = 0.0
      
      logger.info("***** Running training *****")
      
      self.model.train()
      for batch_idx, x in enumerate(train_dataloader):
         x = x.to(self.device)
         
         if self.args.total_anneal_steps > 0:
            anneal = min(self.args.anneal_cap, 1. * self.update_count / self.args.total_anneal_steps)
         else:
            anneal = self.args.anneal_cap
         
         self.update_count+=1

         self.optimizer.zero_grad()
         recon_batch, mu, logvar = self.model(x)

         loss = self.model.loss_function(
             recon_batch, x, mu, logvar, anneal)
         loss.backward()
         train_loss += loss.item()
         self.optimizer.step()
         
         if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
            print('[Train] | Epoch: {:3d} | Batch: {:4d}/{:4d} | Loss: {:4.2f} | Anneal: {:.4f}'.format(
               self.epochs+1,
               batch_idx, len(train_dataloader),
               train_loss / batch_idx, 
               anneal,
               ))
               
      writer.add_scalar('loss/train', 
                        train_loss, 
                        self.epochs)

   def evaluate(self, tr_dataloader, te_dataset, mode='validation'):
      n100_list = []
      r20_list = []
      r50_list = []
      
      total_loss = 0.0
      start_idx = 0
      
      logger.info("***** Running evaluation on %s dataset *****", mode)

      self.model.eval()
      with torch.no_grad():
         for batch_idx, x in enumerate(tr_dataloader):
            x = x.to(self.device)
            end_idx = min(start_idx + self.args.batch_size,
                          te_dataset.shape[0])
            heldout_data = te_dataset[start_idx:end_idx]

            recon_batch, mu, logvar = self.model(x)

            loss = self.model.loss_function(recon_batch, x, mu, logvar)
            total_loss += loss.item()

            # Exclude examples from training and validation (if any)
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[x.cpu().numpy().nonzero()] = -np.inf
            
            n_100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r_20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r_50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n_100)
            r20_list.append(r_20)
            r50_list.append(r_50)
            
            start_idx += self.args.batch_size
            
      total_loss /= len(tr_dataloader)
      n100_list = np.concatenate(n100_list)
      r20_list = np.concatenate(r20_list)
      r50_list = np.concatenate(r50_list)
   

      if mode=='validation':
         
         n100 = np.mean(n100_list)
         r20 = np.mean(r20_list)
         r50 = np.mean(r50_list)
         
         writer.add_scalar('loss/val', total_loss, self.epochs)
         writer.add_scalar('metric/ndcg@100', n100, self.epochs)
         writer.add_scalar('metric/recall@20', r20, self.epochs)
         writer.add_scalar('metric/recall@50', r50, self.epochs)
         
         print("[Valid] | Epoch: {:3d} | Loss: {:4.2f} | NDCG@100: {:5.3f} | Recall@20: {:5.3f} | Recall@50: {:5.3f}".format(
                   self.epochs+1, total_loss, n100, r20, r50))
         print('-'*89)

      self.epochs += 1
      return total_loss, n100_list, r20_list, r50_list
   
   def save_model(self):
      # Save model checkpoint (Overwrite)
      if not os.path.exists(self.args.ckpt_dir):
         os.mkdir(self.args.ckpt_dir)

      # Save argument
      torch.save(self.args, os.path.join(self.args.ckpt_dir, 'args.pt'))
      
      # Save model for inference
      torch.save(self.model.state_dict(), os.path.join(
         self.args.ckpt_dir, 'model.pt'))
      logger.info("Saving model checkpoint to {}".format(
         os.path.join(self.args.ckpt_dir, 'model.pt')))
      
   def load_model(self):
      # Check whether model exists
      if not os.path.exists(self.args.ckpt_dir):
         raise Exception("Model doesn't exists! Train first!")

      try:
         self.args = torch.load(os.path.join(self.args.ckpt_dir, 'args.pt'))
         logger.info("***** Args loaded *****")
         self.model.load_state_dict(torch.load(os.path.join(self.args.ckpt_dir, 'model.pt')))
         self.model.to(self.device)
         logger.info("***** Model Loaded *****")
      except:
         raise Exception("Some model files might be missing...")
