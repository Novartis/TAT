"""
Copyright 2024 Novartis Institutes for BioMedical Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from noam import NoamOpt

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from cluster_split import cluster_split
from TransformerMKIIRegressor_PT import TransformerMkII, TNetMkII
from skorch.callbacks.regularization import GradientNormClipping

# --- load features
WORK_DIR = '../work'
TARGET_DIR = WORK_DIR + '/targets'
r2s = []
r2s_index = []

# --- load preprocessed data

X = np.load(WORK_DIR + '/all_images_all_pool.npy')
nvps = pd.read_csv(WORK_DIR  + '/smiles.csv', index_col = 0)
n_mols = len(nvps)

# --- define hold out set

# need a dictionary from mol name to mol index (because the data set is indexed with integers)
mol_to_index = {}
index_to_mol = {}
cmpds = nvps.index.tolist()
for i,mol in enumerate(cmpds):
    mol_to_index[mol] = i
    index_to_mol[i] = mol


fraction = 0.75
partitions= cluster_split(nvps, 'smiles', fraction, "Auto", False, False)
train = [mol_to_index[x] for x in partitions[0]['train']]
test = [mol_to_index[x] for x in partitions[0]['test']]


feature = 'Nuclei_AreaShape_Zernike_8_2' # should be the same as in preprocess.py
y = np.load(TARGET_DIR  + '/all_images_all_'+ feature +'_pool.npy')



# define model input dimensions
seq_dim = X.shape[-2]
feat_dim = X.shape[-1]
n_seq = X.shape[1]
input_size = feat_dim
max_seq_length = seq_dim

SEQ_DIM_PADDED = seq_dim #max of above seq_dim values
FEATURE_DIM_PADDED = feat_dim #max of above feat_dim values



# --- define some scoring functions

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def r2(y_pred: np.ndarray, y_true: np.ndarray):
    x = y_pred
    y = y_true
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return corr*corr

scoring_funcs = {"mse": mse,
                 "r2": r2,
                 }

# --- modeling part 


net = TNetMkII
model = TransformerMkII

# basic parameters
p = {
    'optimizer': 'noam',
    'lr': 0.00,
    'batch_size': 180,
    'max_epochs': 36, #36
    'module__n_modules': 1,
    'module__layer_size': 512, # dimension of transformer embedding 
    'module__n_heads': 4,
    'module__drop': 0, 
    'module__output_size': 1,
    'optimizer__factor': 0.05, 
    'optimizer__warmup_factor': 3,
    'optimizer__weight_decay':0.0005,
    }        

p['module'] = model
p['module__input_size'] = input_size   
p['module__max_seq_length'] = max_seq_length


# optimizer parameters
betas=(0.9, 0.98) # adam betas
eps=1e-9
criterion= nn.MSELoss

p['criterion']=criterion
p['optimizer'] = NoamOpt
p['optimizer__betas']=betas
p['optimizer__eps']=eps
# coarse calc
n_steps_per_epoch = int((X.shape[0] * X.shape[1] * 0.75) / p['batch_size'])
p['optimizer__warmup'] = n_steps_per_epoch * p['optimizer__warmup_factor'] 
del p['optimizer__warmup_factor']
p['optimizer__model_size'] = p['module__layer_size']

# hardware
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
p['device']=device

# gradient clipping as skorch callbacks

clip_norm = 1
clipper = GradientNormClipping(gradient_clip_value=clip_norm)

callbacks_list = [("clipper", clipper)]
p['callbacks']= callbacks_list

# misc parameters
p['iterator_train__shuffle']=True
p['iterator_train__num_workers']=24
p['iterator_train__drop_last']=True
p['train_split']=None


# --- setup data

X_train = X[train].reshape(-1,max_seq_length,input_size).astype(np.float32)
y_train = y[train].reshape(-1,1).astype(np.float32) 
X_test = X[test].reshape(-1,max_seq_length,input_size).astype(np.float32)
y_test = y[test].reshape(-1,1).astype(np.float32)

estimator_params = p
estimator = net(**estimator_params)

# fit on train
estimator.fit(X_train,y_train)

# predict on test set
preds = estimator.predict(X_test)
GT = y_test[:,0].squeeze()


scores = defaultdict(list)
preds_orig = preds
GT_orig = GT
for scorer, func in scoring_funcs.items():
    # summarize per compound
    preds = preds_orig
    GT = GT_orig
    data = pd.DataFrame(np.stack([preds.squeeze(), GT], axis=1), columns=['preds', 'GT'])
    data_per_cmpd_mean = data.groupby(data.index // n_seq).mean()
    preds = data_per_cmpd_mean['preds'].values
    GT = data_per_cmpd_mean['GT'].values

    # plot
    fig = plt.figure()
    sns.scatterplot(data_per_cmpd_mean,x='GT', y='preds')
    plt.savefig(WORK_DIR + '/scatter_' + str(feature) + '.pdf', dpi=300,bbox_inches = "tight")
    
    # compute value
    scores[scorer].append(
        func(preds.squeeze(), GT)
    )
    print(scorer + ": " + str(scores[scorer]))
    if scorer == 'r2':
        r2s.append(scores[scorer][0])
        r2s_index.append(feature)
        #print(r2s)


r2_df = pd.DataFrame(r2s, index = r2s_index)
r2_df.to_csv(WORK_DIR + '/r2s.csv')

