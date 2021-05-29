import os 
import argparse
import numpy as np
import pytorch_lightning as pl

from data import MovieLens_DataModule
from models import MultiVAE
from models import MultiDAE
from metric import Recall_at_k_batch, NDCG_binary_at_k_batch
from trainer import Trainer


def cli_main(args):
    pl.seed_everything(args.seed)
    
    movielens_dm = MovieLens_DataModule(args)
    if os.path.isdir('./data/ml-20m/pro_sg')==False:
        movielens_dm.setup()

    train_data = movielens_dm.load_data(stage="train")
    val_tr, val_te = movielens_dm.load_data(stage="validation")
    test_tr, test_te = movielens_dm.load_data(stage="test")
    n_items = movielens_dm.load_n_items()

    p_dims = [200, 600, n_items]
    trainer = Trainer(args, p_dims)
    
    best_ndcg = -np.inf
    for epoch in range(args.epochs):
        trainer.train(train_data)
        _, n100_list, r20_list, r50_list = trainer.evaluate(
            val_tr, val_te, mode="validation")

        result = np.mean(n100_list)
        if result > best_ndcg:
            trainer.save_model()
            best_ndcg = result
    
    trainer.load_model()
    test_loss, n100_list, r20_list, r50_list  = trainer.evaluate(test_tr, test_te, mode="test")
    print("-"*89)
    print("[FINAL RESULT]")
    print("Test Loss= %.5f" % (test_loss))
    print("Test NDCG@100=%.5f (%.5f)" %
          (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" %
        (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" %
        (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Version Variational Autoencoders for Collaborative Filtering")
    parser.add_argument("--data_dir", default="./data/ml-20m",
                        type=str, help="The input data dir")
    parser.add_argument("--data_url", default="https://files.grouplens.org/datasets/movielens/ml-20m.zip",
                        type=str, help="Download File URL")
    parser.add_argument("--ckpt_dir", default="./ckpt",
                        type=str, help="Path for saving model")
    parser.add_argument("--model_name", default="multi-vae", 
                        type=str, help="Model type selected in the [multi-vae, mutli-dae]:")
    
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay coefficient')
    parser.add_argument("--batch_size", type=int, default=500,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="upper epoch limit")
    parser.add_argument("--total_anneal_steps", type=int, default=200000,
                        help="the total number of gradient updates for annealing")
    parser.add_argument("--anneal_cap", type=float, default=0.2,
                        help="largest annealing parameter")
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    
    parser.add_argument("--seed", type=int, default=98765,
                        help="random seed")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="num workers for dataloader")
    
    args = parser.parse_args()
    cli_main(args)
