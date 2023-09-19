#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.sparse as sp
import os
import random
import torch

def process_adj(edges,n,loop=False,normalize_adj=False):
    m=edges.shape[0]
    u,v=edges[:,0],edges[:,1]
    adj=sp.coo_matrix((np.ones(m),(u,v)),shape=(n,n),dtype=np.float32)
    adj+=adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)
    if loop:
        adj += sp.eye(n)
    if normalize_adj:
        degrees=np.power(np.array(np.sum(adj, axis=1)),-0.5).flatten()
        degrees=sp.diags(degrees)
        adj=(degrees.dot(adj.dot(degrees)))
    return adj

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
