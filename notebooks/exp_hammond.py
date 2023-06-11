import numpy as np
from data import load_beeline
from logger import LightLogger
from runner import runGRNVAE, runGRNVAE_ensemble
from runner import DEFAULT_DEEPSEM_CONFIGS, DEFAULT_GRNVAE_CONFIGS
from evaluate import extract_edges, get_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
import scanpy as sc
import datetime
import sys
from sklearn.metrics import average_precision_score, roc_auc_score


data_name = sys.argv[1]
save_dir = 'final_hammond_remove_gene_first_complete'

hammond_dir = 'data/other_data/Hammond_processed/'
# with open(f'{hammond_dir}/top10000_features_all.txt') as f:
#     vari_genes = [x.rstrip() for x in f.readlines()]
# vari_genes = [x for x in vari_genes if not x.lower().startswith('mt') and not x.lower().startswith('rps') and not x.lower().startswith('rpl') and not x.lower().startswith('gm')]
# vari_genes = set(vari_genes)

# data = sc.read(f'{hammond_dir}/final/{data_name}_data.csv')
# data = data.transpose()
# data_X = data.X
# n_obs, n_gene = data_X.shape
# gene_names = data.var_names
# vari_gene_idx = [x in vari_genes for x in data.var_names]
# data_X = data.X[:, vari_gene_idx]
# n_obs, n_gene = data_X.shape
# gene_names = data.var_names[vari_gene_idx]

def gene_filter(x):
    x = x.lower()
    return (not (x.startswith(('mt', 'rps', 'rpl', 'gm', 'adap2os'))))

data = sc.read(f'{hammond_dir}/final/{data_name}_data.csv')
data = data.transpose()

zero_genes = set(data.var_names[data.X.mean(0) == 0].to_list())
valid_genes = [x for x in data.var_names if gene_filter(x) and x not in zero_genes]
valid_gene_idx = [x in valid_genes for x in data.var_names]
data_X = data.X[:, valid_gene_idx]
n_obs, n_gene = data_X.shape
gene_names = data.var_names[valid_gene_idx]
with open(f'results/{save_dir}/validgenes_{data_name}.txt', 'w') as f:
    f.writelines('\n'.join(valid_genes))

logger = LightLogger()
vae, _ = runGRNVAE(
    data_X, DEFAULT_GRNVAE_CONFIGS, logger=logger, 
    progress_bar=True)

# m = vae.cuda()

# x = torch.tensor(data_X).cuda()
# x_g_mean = torch.tensor(data_X.mean(0)).cuda()
# x_g_std = torch.tensor(data_X.std(0)).cuda()

# model_output = m(x, x_g_mean, x_g_std)


# da_pred = torch.sigmoid(model_output['da_pred']).detach().cpu().numpy()
# da_pred = da_pred.flatten()[(data_X != 0).flatten()]
# da_target = model_output['da_mask'].detach().cpu().numpy()
# da_target = da_target.flatten()[(data_X != 0).flatten()]
# auroc = roc_auc_score(da_target, da_pred)
# aupr = average_precision_score(da_target, da_pred)
# print(f'AUROC: {auroc} AUPR: {aupr}')

# model_output_no_da = m(x, x_g_mean, x_g_std, use_dropout_augmentation=False)
# x_rec = (model_output_no_da['x_rec'] * x_g_std + x_g_mean).detach().cpu().numpy()
# da_prob = torch.sigmoid(model_output_no_da['da_pred']).detach().cpu().numpy()

now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
np.save(f'results/{save_dir}/adjmatrix_{data_name}_{now_time}.npy', vae.get_adj())
extract_edges(vae.get_adj(), gene_names).to_csv(f'results/{save_dir}/edgelist_{data_name}_{now_time}.csv', index=False)
# np.save(f'results/final_hammond/xrec_{data_name}_{now_time}.npy', x_rec)
# np.save(f'results/final_hammond/daprob_{data_name}_{now_time}.npy', da_prob)
logger.to_df().to_csv(f'results/{save_dir}/logger_{data_name}_{now_time}.csv', index=False)
