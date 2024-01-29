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
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy
import numpy as np
import joblib
from sklearn import cross_decomposition
from sklearn.model_selection import KFold
import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

from collections import namedtuple

# Took this code from https://github.com/Novartis/pQSAR/blob/e05e4e18e4237f3ca611ea9dd81651c7e5cf55cb/CommonTools.py

def cluster_split(data, smiles, fraction2train, clusterMethod="Auto", dropFPs=True, dropMol=True):
    data = data.copy()
    molecule = 'molecule'
    try:
    	PandasTools.AddMoleculeColumnToFrame(data, smiles, molecule)
    except:
    	print("Could not add molecule column to frame")
    
    #remove records with empty molecules
    data = data.loc[data[molecule].notnull()]
    
    data['FP'] = [computeFP(m) for m in data[molecule]]
    
    #filter potentially failed fingerprint computations
    #data = data.loc[data['FP'].notnull()]
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in data[molecule]]
    if dropMol == True:
    	data.drop(molecule, axis=1, inplace=True)	
    #data['FP'] = [computeFP(fp) for fp in fps]
    min_select = int(fraction2train * len(data))
    
    cluster_list = ClusterFps(fps, clusterMethod)
    
    cluster_list.sort(key=len, reverse=True)
    flat_list = sum(cluster_list, [])
    keep_tuple = flat_list[0 : min_select]
    
    if dropFPs == True:
        data.drop('FP', axis=1, inplace=True)
    
    train = data.iloc[list(keep_tuple)].copy()
    test = data.drop(train.index)
    # as in toxsquad
    partitions = []
    partition = {"train": train.index, "test": test.index}
    partitions.append(partition)
    return partitions


def dists_yield(fps, nfps):
    # generator
    for i in range(1, nfps):
        yield [1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])]        

def ClusterData(fps, nPts, distThresh, reordering=False):
    """	clusters the data points passed in and returns the list of clusters
    	**Arguments**
    		- data: a list of items with the input data
    			(see discussion of _isDistData_ argument for the exception)
    		- nPts: the number of points to be used
    		- distThresh: elements within this range of each other are considered
    			to be neighbors
    		- reodering: if this toggle is set, the number of neighbors is updated
    				 for the unassigned molecules after a new cluster is created such
    				 that always the molecule with the largest number of unassigned
    				 neighbors is selected as the next cluster center.
    	**Returns**
    		- a tuple of tuples containing information about the clusters:
    			 ( (cluster1_elem1, cluster1_elem2, ...),
    				 (cluster2_elem1, cluster2_elem2, ...),
    				 ...
    			 )
    			 The first element for each cluster is its centroid.
    """
    nbrLists = [None] * nPts
    for i in range(nPts):
        nbrLists[i] = []
    
    #dmIdx = 0
    dist_fun = dists_yield(fps, nPts)
    for i in range(1, nPts):
    	#print(i)
    	dists = next(dist_fun)
    
    	for j in range(i):
    	    #if not isDistData:
    	    #	dij = EuclideanDist(data[i], data[j])
    	    #else:
    	    	#dij = data[dmIdx]
    	    dij = dists[j]
    	    	#dmIdx += 1
    	    if dij <= distThresh:
                nbrLists[i].append(j)
                nbrLists[j].append(i)
    
    # sort by the number of neighbors:
    tLists = [(len(y), x) for x, y in enumerate(nbrLists)]
    tLists.sort(reverse=True)
    
    res = []
    seen = [0] * nPts
    while tLists:
    	_, idx = tLists.pop(0)
    	if seen[idx]:
    	    continue
    	tRes = [idx]
    	for nbr in nbrLists[idx]:
    	    if not seen[nbr]:
                tRes.append(nbr)
                seen[nbr] = 1
    	# update the number of neighbors:
    	# remove all members of the new cluster from the list of
    	# neighbors and reorder the tLists
    	res.append(tRes)
    return res

def computeFP(x):
    #compute depth-2 morgan fingerprint hashed to 1024 bits
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)
        res = np.zeros(len(fp), np.int8)
        #convert the fingerprint to a numpy array and wrap it into the dummy container
        DataStructs.ConvertToNumpyArray(fp, res)
        return FP(res)
    except:
        #print("FPs for a structure cannot be calculated")
        return None

def ClusterFps(fps, method="Auto"):
    #Cluster size is probably smaller if the cut-off is larger. Changing its values between 0.4 and 0.45 makes a lot of difference
    nfps = len(fps)
    
    if method == "Auto":
        if nfps >= 10000:
    	    method = "TB"
        else:
    	    method = "Hierarchy"
                
    if method == "TB":
    	#from rdkit.ML.Cluster import Butina
    	cutoff = 0.56
    	print("Butina clustering is selected. Dataset size is:", nfps)
    
    	cs = ClusterData(fps, nfps, cutoff)
    	
    elif method == "Hierarchy":
    	print("Hierarchical clustering is selected. Dataset size is:", nfps)
    
    	disArray = pdist(fps, 'jaccard')
    	#Build model
    	Z = hierarchy.linkage(disArray)
    	
    	#Cut-Tree to get clusters
    	#x = hierarchy.cut_tree(Z,height = cutoff)
    	average_cluster_size = 8
    	cluster_amount = int( nfps / average_cluster_size )	 # calculate total amount of clusters by (number of compounds / average cluster size )
    	x = hierarchy.cut_tree(Z, n_clusters = cluster_amount )		#use cluster amount as the parameter of this clustering algorithm. 
    	
    	#change the output format to mimic the output of Butina
    	x = [e[0] for e in x]
    	cs = [[] for _ in set(x)]
    
    	for i in range(len(x)):
    	    cs[x[i]].append(i)
    return cs
