# Variational AutoEncoder For Collaborative Filtering

An Implementation of [Variational Autoencoders for Collaborative Filtering (Liang et al. 2018)](https://arxiv.org/abs/1802.05814) in PyTorch.


## Dependencies
* python 3.8
```{python}
pip install -r requirements.txt
```

## How to run
```{python}
python main.py
```

## Results
**RUN EPOCH 50**

**Test Result**
|         Diff        | Model(unofficial) |   Official Code   |
|:-------------------:|:-----------------:|:-----------------:|
|        Epoch        |         50        |        200        |
|         Loss        |     472.55275     |         -         |
|  NDCG@100 (mean/sd) | 0.42109 / 0.00209 | 0.42592 / 0.00211 |
| Recall@20 (mean/sd) | 0.39145 / 0.00269 | 0.39535 / 0.00270 |
| Recall@50 (mean/sd) | 0.53414 / 0.00285 | 0.53540 / 0.00284 |


![image](https://user-images.githubusercontent.com/47301926/116814490-2c334880-ab94-11eb-8be1-ed3c063c7033.png)


## Reference 
[Official Code Implementation](https://github.com/dawenl/vae_cf)
