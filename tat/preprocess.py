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

import os
from itertools import product, repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd

# --- DATA 
# --- directory for output
WORK_DIR = '../work/'
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)

TARGET_DIR = '../work/targets'
if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)


# --- directory with transcriptomic data (X) and target (y=f(X))
data_dir = '/path/to/rosetta/s3/LINCS-Pilot1/L1000'
data_file = 'level_4.csv.gz'
target_data_file = '/path/to/rosetta/s3/LINCS-Pilot1/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz'

# --- read X data
df = pd.read_csv(data_dir + '/' + data_file, compression='gzip')
original_df = df.copy()

# --- read y data
target_df = pd.read_csv(target_data_file, compression='gzip', low_memory=False)
target_df['cmpd'] = target_df['Metadata_pert_id_dose'].apply(lambda x: x.split('_')[0])
target_df['conc'] = target_df['Metadata_pert_id_dose'].apply(lambda x: x.split('_')[-1])

idx = target_df['conc'] == '10.0' # select concentration level for __target__ 
target_df = target_df.loc[idx]
target_df = target_df.astype({'conc':float})

def get_images_for_compound(count_df, cmpd, sub_df, max_reps = 2): 
    reps = sub_df['Replicate_Number'].unique()
    concs = sub_df['conc_levels'].unique()
    reps.sort()
    concs.sort()

    
    if len(reps) > max_reps:
        reps = tuple(range(1,max_reps+1)) 

    combs = pd.DataFrame(list(product(concs, reps)), columns=['concs', 'reps'])
    n_concs = len(concs)    
    n_reps = len(reps)
    
    sets = []
    for k in range(n_concs):
        sets.append(combs.iloc[ (k*n_reps):(k+1)*n_reps].index.values)

    try:
        rep_combinations = list(product(sets[0],sets[1],sets[2])) # CHANGE depending on number of concentrations

    except:
        return cmpd, None
    
    rep_combs = pd.DataFrame(rep_combinations, columns=concs)
    cmpd_images = []
    FAIL = False
    for seq_idx, seq in rep_combs.iterrows():
        seq_image = []

        for element in seq:
            c = combs.iloc[element]['concs']
            r = combs.iloc[element]['reps']
            idx = np.logical_and(sub_df['conc_levels'] == c ,
                                 sub_df['Replicate_Number'] == r)
            idx = sub_df.loc[idx].index
            feature_vec = count_df.loc[idx].values.squeeze()
            if len(feature_vec) == 0:
                FAIL = True
                break
            seq_image.append(feature_vec)
            
        if FAIL:
            break
        seq_image = np.stack(seq_image)
        
        cmpd_images.append(seq_image)
    if FAIL:
        cmpd_images = None
    return cmpd, cmpd_images


selected_feature = 'Nuclei_AreaShape_Zernike_8_2' # set same feature in model_build.py

# build dict
cmpds = target_df['cmpd'].unique().tolist()
feature_dict = {}
for c in cmpds:
    idx = target_df['cmpd'] == c
    subdf = target_df.loc[idx][selected_feature]
    val = subdf.median() # aggregate over replicate
    feature_dict[c] = val

# --- derive treatments and concentrations
df['cmpd'] = df['pert_id_dose'].apply(lambda x: x.split('_')[0])
df['conc'] = df['pert_id_dose'].apply(lambda x: x.split('_')[-1])


avail_conc = df['conc'].unique()
# --- grab a subset of concentrations somewhat mapping to our concentrations (viz, 0.1, 1, and 10 uM)
# '10', '3.33', '1.11', '0.37', '0.12',  '0.04'
#
selected_conc = ['10', '1.11', '0.12']
dfs = []
for c in selected_conc:
    idx = df['conc'] == c
    dfs.append(df.loc[idx])

df = pd.concat(dfs, axis=0)



# --- drop controls
controls = ['DMSO']
for ctrl in controls:
    idx = df['cmpd'] == ctrl
    df = df.loc[~idx]

# --- define concentration levels
df['conc_levels']=""
cmpds = df['cmpd'].unique().tolist()
for cmpd in cmpds:
    samples_idx = df.index[df['cmpd'] == cmpd]    
    cmpdConc = df.loc[samples_idx]['conc'];
    concLevels,_ = pd.factorize(cmpdConc);
    df.loc[samples_idx,'conc_levels'] = concLevels

# --- add replicate number
df['Replicate_Number'] = df.groupby(['cmpd','conc_levels']).cumcount()+1

# --- check if we have consistent conc_levels
by_cmpd = df.groupby(['cmpd'])['conc_levels'].unique().apply(lambda x: len(x))
#print(by_cmpd.describe()) # the code later on will take care of those that don't have all three concentrations

# --- check if we have consistent replicates
by_cmpd_conc = df.groupby(['cmpd','conc_levels'])['Replicate_Number'].count()
by_cmpd_conc.describe() # most of them have three

# --- remove compounds where concentrations have less we have less than n_reps replicates
n_reps = 3
cmpds_to_remove = by_cmpd_conc.loc[by_cmpd_conc < n_reps].index.tolist()
try:
    cmpds_to_remove = list(set(list(zip(*cmpds_to_remove))[0]))
except Exception as e:
    print(e)

    
for cmpd in cmpds_to_remove:
    idx = df['cmpd'] == cmpd
    df = df.loc[~idx]

# --- create count matrix
df.set_index('cid', inplace=True)
cols = df.columns
dfs = df[cols[:-55]]
meta = df[cols[-55:]] # just the meta

# --- CREATE SEQUENCES (multi-concentration profiles)
nvps = meta['cmpd'].unique()
failed_compounds = 0


cmpd_batch = []
df_batch = []
smiles_dict = {}
target_y_dict = {}


for i, cmpd in enumerate(nvps):
    sub_df = meta.loc[meta['cmpd'] == cmpd]
    cmpd_batch.append(cmpd)
    df_batch.append(sub_df)
    # gather smiles, too
    smiles = sub_df['x_smiles'].unique()[0]
    smiles_dict[cmpd]=smiles
    try:
        target_y = feature_dict[cmpd]
    except:
        target_y = None


    target_y_dict[cmpd]=target_y



n_cpu = 32
pool = Pool(n_cpu)
cmpd_images = pool.starmap(get_images_for_compound, zip(repeat(dfs), cmpd_batch, df_batch, repeat(n_reps)))



# --- put it all together

class_index = 0

all_images = []
ground_truth = []
cmpd_gt_dict = {}
smiles_dict_goodmols = {}
for bundle in cmpd_images:
    cmpd = bundle[0]
    image = bundle[1]
    target_y = target_y_dict[cmpd]
    if image is not None:
        if target_y is not None:
            all_images.append(image)
            cmpd_gt_dict[cmpd] = class_index
            smiles_dict_goodmols[cmpd] = smiles_dict[cmpd]
            # ground-truth is just the target variable to be predicted (e.g., pAC50 of compound in assay)
            ground_truth.extend(np.ones([1,len(image)]) * target_y)
            class_index +=1
        else:
            print("[ERROR] target is none, should not happen")

all_images = np.stack(all_images)
ground_truth = np.stack(ground_truth)

# --- SAVE RESULTS

np.save(WORK_DIR + '/all_images_all_pool.npy', all_images)
np.save(WORK_DIR + '/targets/all_images_all_' + selected_feature +'_pool.npy', ground_truth)

#
cmpd_gt_dict = pd.DataFrame.from_dict(cmpd_gt_dict, orient='index', columns=['class_id'])
cmpd_gt_dict.to_csv(WORK_DIR + '/classes_all_pool.csv')

# save smiles, too, for the cluster-based split
smiles = pd.DataFrame.from_dict(smiles_dict_goodmols, orient='index',columns=['smiles'])
smiles.to_csv(WORK_DIR + '/smiles.csv')

    



