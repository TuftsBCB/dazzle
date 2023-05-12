import numpy as np
from data import load_beeline
from logger import LightLogger
# from runner import runGRNVAE, runGRNVAE_ensemble, DEFAULT_GRNVAE_CONFIGS
# from runner import runDeepSEM, runDeepSEM_ensemble, DEFAULT_DEEPSEM_CONFIGS
from evaluate import extract_edges, get_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
import scanpy as sc
import sys
import datetime

da_p = float(sys.argv[1])
da_t = sys.argv[2]
non_zero = (sys.argv[3] == 'nz')
delay = int(sys.argv[4])
chi = float(sys.argv[5])

now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()

    def forward(self, x):
        out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1)) 
        return self.l3(out2)

class GRNVAE(nn.Module):
    def __init__(
        self, n_gene, hidden_dim=128, z_dim=1, A_dim=1, 
        activation=nn.Tanh, train_on_non_zero=False, 
        dropout_augmentation_p=0.05, dropout_augmentation_type='enhance',
        pretrained_A=None, 
    ):
        super(GRNVAE, self).__init__()
        self.n_gene = n_gene
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.A_dim = A_dim
        self.train_on_non_zero = train_on_non_zero
        
        if pretrained_A is None:
            adj_A = torch.ones(A_dim, n_gene, n_gene) / (n_gene - 1) 
            adj_A += torch.rand_like(adj_A) * 0.0002
        else:
            adj_A = pretrained_A
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        
        self.inference_zposterior = MLP(1, hidden_dim, z_dim * 2, activation)
        self.generative_pxz = MLP(z_dim, hidden_dim, 1, activation)
        self.da_classifier = MLP(z_dim, z_dim, 1, activation)
        self.da_p = dropout_augmentation_p
        self.da_type = dropout_augmentation_type
        if self.da_p != 0:
            self.classifier_pos_weight = torch.FloatTensor(
                [(1-dropout_augmentation_p) / dropout_augmentation_p]
            )
        else:
            self.classifier_pos_weight = torch.FloatTensor([1.0])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def get_adj_(self):
        eye_tensor = torch.eye(
            self.n_gene, device = self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        mask = torch.ones_like(self.adj_A) - eye_tensor
        return (self.adj_A * mask).mean(0)
    
    def get_adj(self):
        return self.get_adj_().cpu().detach().numpy()
    
    def I_minus_A(self):
        eye_tensor = torch.eye(
            self.n_gene, device = self.adj_A.device
        ).repeat(self.A_dim, 1, 1)
        # clean up A along diagnal line
        mask = torch.ones_like(self.adj_A) - eye_tensor
        clean_A = self.adj_A * mask
        return eye_tensor - clean_A
    
    def reparameterization(self, z_mu, z_sigma):
        return z_mu + z_sigma * torch.randn_like(z_sigma)
    
    @torch.no_grad()
    def dropout_augmentation(self, x, global_mean):
        da_mask = (torch.rand_like(x) < self.da_p)
        if self.da_type == 'belowmean':
            da_mask = da_mask * (x < global_mean)
        elif self.da_type == 'belowhalfmean':
            da_mask = da_mask  * (x < (global_mean / 2))
        elif self.da_type == 'all':
            da_mask = da_mask
        noise = -1 * x * da_mask
        x = x + noise
        return x, noise, da_mask
        
    def forward(self, x, global_mean, global_std):                
        if self.train_on_non_zero:
            non_zero_mask = (x != 0)
        else:
            non_zero_mask = torch.ones_like(x)
        
        x_init = x
        x, noise, da_mask = self.dropout_augmentation(x_init, global_mean)
        
        x = (x - global_mean) / (global_std)
        noise = (noise - global_mean) / (global_std)
        
        # Encoder --------------------------------------------------------------
        I_minus_A = self.I_minus_A()
                
        z_posterior = self.inference_zposterior(x.unsqueeze(-1))
        z_posterior = torch.einsum('ogd,agh->ohd', z_posterior, I_minus_A)
        z_mu = z_posterior[:, :, :self.z_dim]
        z_logvar = z_posterior[:, :, self.z_dim:]
        z = self.reparameterization(z_mu, torch.exp(z_logvar * 0.5))
        # z = torch.einsum('ogd,og->ogd', z, (~da_mask))
        
        # Decoder --------------------------------------------------------------
        z_inv = torch.einsum('ogd,agh->ohd', z, torch.inverse(I_minus_A))
        x_rec = self.generative_pxz(z_inv).squeeze(2)
        
        # DA classifier
        da_pred = self.da_classifier(z).squeeze(-1)
        loss_da = F.binary_cross_entropy_with_logits(
            da_pred, da_mask.float(), pos_weight=self.classifier_pos_weight)
        
        # Losses ---------------------------------------------------------------
        if self.train_on_non_zero:
            eval_mask = non_zero_mask 
        else:
            eval_mask = torch.ones_like(x)
            
        loss_rec_all = (x - x_rec).pow(2)
        loss_rec = torch.sum(loss_rec_all * eval_mask)
        loss_rec = loss_rec / torch.sum(eval_mask)
        
        loss_kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar))
                        
        out = {
            'loss_rec': loss_rec, 'loss_kl': loss_kl, 
            'z_posterior': z_posterior, 'z': z, 
            'da_pred': da_pred, 'da_mask': da_mask, 'loss_da': loss_da, 
        }
        return out

    
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
# from models import GRNVAE, GRNVAE_3dA
from evaluate import get_metrics
from tqdm import tqdm
from logger import LightLogger

DEFAULT_DEEPSEM_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 0,
    'train_on_non_zero': False,
    'dropout_augmentation': 0.0,
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'h_scale': 0,
    'delayed_steps_on_sparse': 0,
    
    # Neural Net Training
    'batch_size': 64,
    'n_epochs': 120,
    'schedule': [1000],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 1
}

DEFAULT_GRNVAE_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 0,
    'train_on_non_zero': True,
    'dropout_augmentation': 0.5,
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'h_scale': 0,
    'delayed_steps_on_sparse': 10,
    
    # Neural Net Training
    'batch_size': 128,
    'n_epochs': 64,
    'schedule': [120, 240],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 1
}

def one_hot(x):
    x_unique = np.unique(x)
    n_obs = x.shape[0]
    n_classes = x_unique.shape[0]
    
    label_dict = {label: i for i, label in enumerate(x_unique)}
    
    one_hot_matrix = np.zeros([n_obs, n_classes])
    for i, label in enumerate(x):
        one_hot_matrix[i, label_dict[label]] = 1.0
    return one_hot_matrix, x_unique

def runGRNVAE(exp_array, configs, 
              ground_truth=None, logger=None):
    '''
    Initialize and Train a GRNVAE model with configs
    
    Parameters
    ----------
    exp_array: np.array
        Expression data with cells on rows and genes on columns. 
    configs: dict
        A dictionary defining various hyperparameters of the 
        model. See Hyperparameters include `train_split`, 
        `train_split_seed`, `batch_size`, `hidden_dim`, `z_dim`,
        `train_on_non_zero`, `dropout_augmentation`, `cuda`,
        `alpha`, `beta`, `delayed_steps_on_sparse`, `n_epochs`, 
        `eval_on_n_steps`, `lr_nn`, `lr_adj`, `K1`, and `K2`. 
    ground_truth: tuple or None
        (Optional, only for BEELINE evaluation) You don't need 
        to define this parameter when you execute GRNVAE on real 
        datasets when the ground truth network is unknown. For 
        evaluations on BEELINE, 
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. The first element of this
        tuple is eval_flat_mask, the boolean mask on the flatten
        adjacency matrix to identify TFs and target genes. The
        second element is the lable values y_true after flatten.
    logger: LightLogger or None
        Either a predefined logger or None to start a new one. This 
        logger contains metric information logged during training. 
        
    Returns
    -------
    torch.Module
        A GRNVAE module object. You can export the adjacency matrix
        using its get_adj() method. 
    '''
    if configs['early_stopping'] != 0 and configs['train_split'] == 1.0:
        raise Exception(
            "You indicate early stopping but you have not specified any ", 
            "validation data. Consider decrease your train_split. ")
    es = configs['early_stopping']
    
    n_obs, n_gene = exp_array.shape
        
    # Logger -------------------------------------------------------------------
    if logger is None:
        logger = LightLogger()
    logger.set_configs(configs)
    note_id = logger.start()

    # cell_min = exp_array.min(1, keepdims=True)
    # cell_max = exp_array.max(1, keepdims=True)
    # exp_array = (exp_array - cell_min) / (cell_max - cell_min)
    
    # Global Mean/Std ----------------------------------------------------------
    global_mean = torch.FloatTensor(exp_array.mean(0))
    global_std = torch.FloatTensor(exp_array.std(0))

    # Train/Test split if requested --------------------------------------------
    assert configs['train_split']>0 and configs['train_split']<=1, \
        f'Expect 0<configs["train_split"]<=1'
    has_train_test_split = (configs['train_split'] != 1.0)
    
    if configs['train_split_seed'] is None:
        train_mask = np.random.rand(n_obs)
    else:
        rs = np.random.RandomState(seed=configs['train_split_seed'])
        train_mask = rs.rand(n_obs)
        
    train_dt = TensorDataset(
        torch.FloatTensor(exp_array[train_mask <= configs['train_split'], ]),
    )
    train_loader = DataLoader(
        train_dt, batch_size=configs['batch_size'], shuffle=True)
    if has_train_test_split:
        val_dt = TensorDataset(
            torch.FloatTensor(exp_array[train_mask > configs['train_split'], ]),
        )
        val_loader = DataLoader(
            val_dt, batch_size=configs['batch_size'], shuffle=True)

    # Defining Model -----------------------------------------------------------
    vae = GRNVAE(
        n_gene = n_gene, 
        hidden_dim=configs['hidden_dim'], z_dim=configs['z_dim'], 
        A_dim = configs['A_dim'],
        train_on_non_zero=configs['train_on_non_zero'], 
        dropout_augmentation_p=configs['dropout_augmentation_p'],
        dropout_augmentation_type=configs['dropout_augmentation_type']
        # A_dim=configs['A_dim']
    )
    
    # Move things to cuda if necessary -----------------------------------------
    if configs['cuda']:
        global_mean = global_mean.cuda()
        global_std = global_std.cuda()
        vae = vae.cuda()
        vae.classifier_pos_weight = vae.classifier_pos_weight.cuda()
    
    if configs['number_of_opt'] == 2:
        opt_nn = torch.optim.RMSprop(vae.parameters(), lr=configs['lr_nn'])
        opt_adj = torch.optim.RMSprop([vae.adj_A], lr=configs['lr_adj'])
        scheduler_nn = torch.optim.lr_scheduler.StepLR(
            opt_nn, step_size=configs['schedule'][0], gamma=0.5)
    else:
        param_all_but_adj = [p for i, p in enumerate(vae.parameters()) if i != 0]
        param_adj = [vae.adj_A]
        opt = torch.optim.Adam([{'params': param_all_but_adj}, 
                                   {'params': param_adj}], 
                                  lr=configs['lr_nn'], 
                               weight_decay=0.00, betas=[0.9, 0.9]
                              )
        # opt = torch.optim.RMSprop([{'params': param_all_but_adj}, 
        #                            {'params': param_adj}], 
        #                           lr=configs['lr_nn'], 
        #                        weight_decay=0.00
        #                       )
        opt.param_groups[0]['lr'] = configs['lr_nn']
        opt.param_groups[1]['lr'] = configs['lr_adj']
        opt.param_groups[1]['weight_decay'] = 0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=configs['schedule'], gamma=0.5)
        
    # Training loops -----------------------------------------------------------
    es_tracks = []
    adjs = []
    for epoch in range(configs['n_epochs']):
        if configs['number_of_opt'] == 2:
            vae.train(True)
            iteration_for_A = epoch%(configs['K1']+configs['K2'])>=configs['K1']
            vae.adj_A.requires_grad = iteration_for_A
            evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)

            # go through training samples 
            eval_log = {
                'train_loss_rec': 0, 'train_loss_kl': 0, 'train_loss_sparse': 0, 
                'train_loss_da': 0, 'train_loss': 0
            }
            for i, batch in enumerate(train_loader):
                exp = batch[0]
                if configs['cuda']:
                    exp = exp.cuda()

                if iteration_for_A:
                    opt_adj.zero_grad()
                    out = vae(exp, global_mean, global_std)
                else:
                    opt_nn.zero_grad()
                    out = vae(exp, global_mean, global_std)
                loss = out['loss_rec'] + configs['beta'] * out['loss_kl'] 
                adj_m = vae.get_adj_()
                loss_sparse = torch.norm(adj_m, 1) / n_gene / n_gene
                if epoch >= configs['delayed_steps_on_sparse']:
                    loss += configs['alpha'] * loss_sparse
                loss.backward()
                if iteration_for_A:
                    opt_adj.step()
                else:
                    opt_nn.step()
                scheduler_nn.step()
                eval_log['train_loss_rec'] += out['loss_rec'].detach().cpu().item()
                eval_log['train_loss_kl'] += out['loss_kl'].detach().cpu().item()
                eval_log['train_loss_sparse'] += loss_sparse.detach().cpu().item()
                eval_log['train_loss'] += loss.detach().cpu().item()
        else:
            vae.train(True)
            evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)

            # go through training samples 
            eval_log = {
                'train_loss_rec': 0, 'train_loss_kl': 0, 'train_loss_sparse': 0,
                'train_loss_da': 0, 'train_loss': 0, 
            }
            for i, batch in enumerate(train_loader):
                exp = batch[0]
                if configs['cuda']:
                    exp = exp.cuda()

                opt.zero_grad()
                out = vae(exp, global_mean, global_std)

                loss = out['loss_rec'] + configs['beta'] * out['loss_kl'] 
                adj_m = vae.get_adj_()
                loss_sparse = torch.norm(adj_m, 1) / n_gene / n_gene
                
                if epoch >= configs['delayed_steps_on_sparse']:
                    loss += configs['alpha'] * loss_sparse
                if configs['dropout_augmentation_p'] != 0:
                    loss += configs['chi'] * out['loss_da']
                loss.backward()
                opt.step()
                scheduler.step()
                eval_log['train_loss_rec'] += out['loss_rec'].detach().cpu().item()
                eval_log['train_loss_kl'] += out['loss_kl'].detach().cpu().item()
                eval_log['train_loss_da'] += out['loss_da'].detach().cpu().item()
                eval_log['train_loss_sparse'] += loss_sparse.detach().cpu().item()
                eval_log['train_loss'] += loss.detach().cpu().item()
        
        for log_item in eval_log.keys():
            eval_log[log_item] /= (i+1)
        
        # go through val samples
        if evaluation_turn:
            adj_matrix = adj_m.cpu().detach().numpy()
            adjs.append(adj_matrix)
            eval_log['negative_adj'] = int(np.sum(adj_matrix < -1e-5))
            if ground_truth is not None:
                epoch_perf = get_metrics(adj_matrix, ground_truth)
                for k in epoch_perf.keys():
                    eval_log[k] = epoch_perf[k]
            
            if has_train_test_split:
                eval_log['val_loss_rec'] = 0
                eval_log['val_loss_kl'] = 0
                eval_log['val_loss_sparse'] = 0
                vae.train(False)
                for batch in val_loader:
                    x = batch[0]
                    if configs['cuda']:
                        x = x.cuda()
                    out = vae(x, global_mean, global_std, 
                              use_dropout_augmentation=False)
                    eval_log['val_loss_rec'] += out['loss_rec'].detach().cpu().item()
                    eval_log['val_loss_kl'] += out['loss_kl'].detach().cpu().item()
                    eval_log['val_loss_sparse'] += out['loss_sparse'].detach().cpu().item()
                if epoch >= configs['delayed_steps_on_sparse']:
                    es_tracks.append(eval_log['val_loss_rec'])
            
            logger.log(eval_log)
            # early stopping
            if (es > 0) and (len(es_tracks) > (es + 2)):
                if min(es_tracks[(-es-1):]) < min(es_tracks[(-es):]):
                    print('Early stopping triggered')
                    break
    logger.finish()
    return vae.cpu(), adjs

def runGRNVAE_ensemble(exp_array, configs,
                       ground_truth=None, logger=None, rep_times=10):
    trained_models = []
    final_adjs = []
    for _ in tqdm(range(rep_times)):
        vae, adjs = runGRNVAE(exp_array, configs, ground_truth, logger)
        trained_models.append(vae)
        final_adjs.append(vae.get_adj())
    ensembled_adj = sum(final_adjs)
    return trained_models, ensembled_adj


# def runGRNVAE_ensemble(exp_array, configs,
#                        ground_truth=None, logger=None, 
#                        dropout_augmentation=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
#     print(f'Running ensemble with following DA: {dropout_augmentation}')
#     trained_models = []
#     final_adjs = []
#     for da in dropout_augmentation:
#         configs['dropout_augmentation'] = da
#         vae, adjs = runGRNVAE(exp_array, configs, ground_truth, logger)
#         trained_models.append(vae)
#         final_adjs.append(vae.get_adj())
#     ensembled_adj = np.prod(np.stack(final_adjs), axis=0)
#     return trained_models, ensembled_adj

DEFAULT_GRNVAE_CONFIGS = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 1,
    'train_on_non_zero': non_zero,
    'dropout_augmentation_p': da_p,
    'dropout_augmentation_type': da_t,
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'chi': chi,
    'h_scale': 0,
    'delayed_steps_on_sparse': delay,
    
    # Neural Net Training
    'number_of_opt': 1,
    'batch_size': 64,
    'n_epochs': 120,
    'schedule': [1000, 2000],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 2
}


logger = LightLogger()
configs = DEFAULT_GRNVAE_CONFIGS
all_scores = []
for bm in ['500_STRING', '1000_STRING', '500_Non-ChIP', '1000_Non-ChIP']:
    for dt in ['hESC', 'hHep', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']:
        data, ground_truth = load_beeline(
            data_dir='data', 
            benchmark_data=dt, 
            benchmark_setting=bm
        )
        configs['dt'] = dt
        configs['bm'] = bm
        
        final_adjs = []
        for i in tqdm(range(10)):
            vae = runGRNVAE(
                data.X, DEFAULT_GRNVAE_CONFIGS, ground_truth, logger
            )
            final_adjs.append(vae[0].get_adj())
            
        ensemble_adj = sum(final_adjs)
        results = get_metrics(ensemble_adj, ground_truth)
        results['dt'] = dt
        results['bm'] = bm
        all_scores.append(results)
        print(f'{bm}-{dt}: {results}') 

import pandas as pd
logger.to_df().to_csv(f'results/0331/logger_{delay}_{non_zero}_{da_t}_{da_p}_{chi}_{now_time}.csv', index=False)
pd.DataFrame(all_scores).to_csv(f'results/0331/ensemble_{delay}_{non_zero}_{da_t}_{da_p}_{chi}_{now_time}.csv', index=False)