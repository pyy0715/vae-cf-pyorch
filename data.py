import os
import sys 
import pandas as pd
import numpy as np
from scipy import sparse

import torch
import pytorch_lightning as pl

from download import download_extract


def filter_triplets(tp, min_uc=5, min_sc=0):
   """
   Args:
      tp ([DataFrame]): [Movielens Dataset]
      min_uc (int, optional): [users who clicked on at least min_uc items]. Defaults to 5.
      min_sc (int, optional): [items which were clicked on by at least min_sc users]. Defaults to 0.
   """
   if min_sc > 0:
      item_count = tp.groupby('movieId').size()
      tp = tp[tp['movieId'].isin(item_count.index[item_count >= min_sc])]
   if min_uc > 0:
      user_count = tp.groupby('userId').size()
      tp = tp[tp['userId'].isin(user_count.index[user_count >= min_uc])]

   user_count, item_count = tp.groupby(
         'userId').size(), tp.groupby('movieId').size()
   return tp, user_count, item_count


def filtering_results(raw_data, user_activity, item_popularity):
    sparsity = 1. * raw_data.shape[0] / \
        (user_activity.shape[0] * item_popularity.shape[0])
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
   
   
def split_users(unique_uid, n_heldout_users=10000):   
   """Split train / validation / test users.
      Select 10K users as heldout users, 10K users as validation users, and the rest of the users for training
   Args:
       unique_uid (Array): [randomly permutated User Index]
       n_heldout_users (int)
   """
   n_users = len(unique_uid)
   tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
   vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
   te_users = unique_uid[(n_users - n_heldout_users):]
   return tr_users, vd_users, te_users


def split_train_test_proportion(data, test_prop=0.2):
   data_grouped_by_user = data.groupby('userId')
   tr_list, te_list = list(), list()
   
   for i, (_, group) in enumerate(data_grouped_by_user):
      n_items_u = len(group)  # per user, item count
      
      if n_items_u >= 5:
         idx = np.zeros(n_items_u, dtype='bool')
         samples = np.random.choice(n_items_u, size=int(
               test_prop * n_items_u), replace=False).astype('int64')
         idx[samples] = True

         tr_list.append(group[np.logical_not(idx)])
         te_list.append(group[idx])
      else:
         tr_list.append(group)

      if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()
            
   data_tr = pd.concat(tr_list)
   data_te = pd.concat(te_list)
   print('Done')
   return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

class MovieLens_DataModule(object):
   def __init__(self, args):
      self.args = args
      self.pro_dir = os.path.join(args.data_dir, 'pro_sg')
      
   def prepare_data(self):
      if not os.path.exists(self.args.data_dir):
         download_extract(self.args.data_url)
      
   def read_data(self, path):
      if path.endswith('csv'):
         return pd.read_csv(path, header=0)
      
   def save_data(self, data, name):
      if not os.path.exists(self.pro_dir):
         os.makedirs(self.pro_dir, exist_ok=True)
         
      if isinstance(data, np.ndarray):
         with open(os.path.join(self.pro_dir, f'{name}.txt'), 'w') as f:
            for i in data:
               f.write('%s\n' % i)
               
      elif isinstance(data, pd.DataFrame):
         data.to_csv(os.path.join(self.pro_dir, f'{name}.csv'), index=False)
         
   def load_n_items(self):
      unique_sid = list()
      with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
         for line in f:
            unique_sid.append(line.strip())
      n_items = len(unique_sid)
      return n_items
         
   def load_data(self, stage=None):
      self.n_items = self.load_n_items()
      
      if stage=='train':
         tp = pd.read_csv(os.path.join(self.pro_dir, 'train.csv'))
         n_users = tp['uid'].max() + 1

         rows, cols = tp['uid'], tp['sid']
         data = sparse.csr_matrix((np.ones_like(rows),(rows, cols)), 
                                  dtype='float64',
                                  shape=(n_users, self.n_items))
         return self.dataloader(data)

      elif stage == 'validation' or stage=='test':
         tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(stage))
         te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(stage))

         tp_tr = pd.read_csv(tr_path)
         tp_te = pd.read_csv(te_path)

         start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
         end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

         rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
         rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

         data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), 
                                     dtype='float64', 
                                     shape=(end_idx - start_idx + 1, self.n_items))
         data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), 
                                     dtype='float64', 
                                     shape=(end_idx - start_idx + 1, self.n_items))
         return self.dataloader(data_tr, shuffle=False), data_te
   
     
   def setup(self):   
      # Read Data, Filter Data   
      self.prepare_data()
      raw_data = self.read_data(path=os.path.join(self.args.data_dir, 'ratings.csv'))
      raw_data = raw_data[raw_data['rating'] > 3.5]
      raw_data, user_activity, item_popularity = filter_triplets(raw_data)
      filtering_results(raw_data, user_activity, item_popularity)
      
      # Shuffle User Index            
      unique_uid = user_activity.index
      idx_perm = np.random.permutation(len(unique_uid))
      unique_uid = unique_uid[idx_perm]
      
      # Split Users
      tr_users, vd_users, te_users = split_users(unique_uid)
      
      train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
      vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
      test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
      
      # Dictionarize the Unique UserId and MovieId
      unique_sid = pd.unique(train_plays['movieId'])
      self.save_data(unique_sid, 'unique_sid') 
      show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
      profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
      
      # Filtering Unique MovieId
      vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
      test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
      
      # Split Dataset
      vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
      test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
      
      # Save Dataset
      train_data = numerize(train_plays, profile2id, show2id)
      self.save_data(train_data, 'train')

      vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
      self.save_data(vad_data_tr, 'validation_tr')

      vad_data_te = numerize(vad_plays_te, profile2id, show2id)
      self.save_data(vad_data_te, 'validation_te')

      test_data_tr = numerize(test_plays_tr, profile2id, show2id)
      self.save_data(test_data_tr, 'test_tr')

      test_data_te = numerize(test_plays_te, profile2id, show2id)
      self.save_data(test_data_te, 'test_te')
      
   def dataloader(self, data, shuffle=False):
      dataset = torch.FloatTensor(data.toarray())
      return torch.utils.data.DataLoader(dataset,
                                         batch_size = self.args.batch_size,
                                         num_workers = 4,
                                         pin_memory = True,
                                         shuffle = shuffle)