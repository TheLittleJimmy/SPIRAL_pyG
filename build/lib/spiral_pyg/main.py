#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import networkx as nx
import time
import random

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.optim import Adam


from .pyG_layers import *
from .utils import *
from pytorch_revgrad import RevGrad

from tqdm import tqdm
import sys



class SPIRAL_integration_pyG:
    def __init__(self,params,samples,feat_file,edge_file,meta_file):
        super(SPIRAL_integration_pyG, self).__init__()
        
        self.params = params
        self.model=A_G_Combination_DA_complex(self.params.AEdims, self.params.AEdimsR,self.params.GSdims,self.params.agg_class,self.params.beta,
                                      self.params.znoise_dim,self.params.CLdims,self.params.DIdims).cuda()
        self.optim=Adam(self.model.parameters(),lr=self.params.lr,weight_decay=self.params.weight_decay)
        self.epochs= self.params.epochs
        self.BS=self.params.batch_size
        self.neighbor_sampler_num=self.params.num_samples
        self.data,self.adj_list,self.feat,self.meta,self.Batch,self.flags=self.load_data(samples,feat_file,edge_file,meta_file)
        
        self.de_act=nn.Sigmoid() 
        self.sample_num=len(np.unique(self.meta.loc[:,'batch']))
        if self.sample_num==2:
            self.cl_act=nn.Sigmoid()
        else:
            self.cl_act=nn.Softmax(dim=1)

        self.FEAT, self.edge_index = self.data.x.cuda(), self.data.edge_index.cuda()
        self.train_loader = NeighborSampler(self.data.edge_index, sizes=self.neighbor_sampler_num, batch_size=self.BS,
                               shuffle=True, num_nodes=self.data.num_nodes,adj_lists=self.adj_list,batch_dict=self.Batch,P=self.params.P,Q=self.params.Q,
                               walks_per_node=self.params.P_WALK,walk_length=self.params.WALK_LEN,negative_walk_length=self.params.N_WALK_LEN,
                                num_negative_samples=self.params.NUM_NEG)
        self.num_postivive_samples=self.params.P_WALK*self.params.WALK_LEN
        self.num_negative_samples=self.params.NUM_NEG
        fix_seed(self.params.random_seed)
          
    def train(self):
        self.model.train()
        print('--------------------------------')
        print('Training.')
        with tqdm(total=self.epochs, file=sys.stdout) as pbar:
            for epoch in np.arange(0,self.epochs):
                total_loss=0.0;AE_loss=0.000;GS_loss=0.000;CLAS_loss=0.000;DISC_loss=0.000
                t=time.time()
                aa=0
                for batch_size, n_id, adjs in self.train_loader:
                    #####forward net######
                    adjs = [adj for adj in adjs]
                    all_embed,ae_out,clas_out,disc_out = self.model(self.FEAT[n_id,:-1], adjs,self.params.lamda,self.de_act,self.cl_act)
                    ae_embed,gs_embed,embed=all_embed
                    [x_bar,x1]=ae_out
                    #####calculate loss######
                    bs=gs_embed.shape[0]//(self.num_postivive_samples+self.num_negative_samples+1)
                    gs_embed, pos_gs_embed, neg_gs_embed = gs_embed.split([bs,self.num_postivive_samples*bs,self.num_negative_samples*bs], dim=0)
                    gs_embed1=gs_embed.repeat(self.num_postivive_samples,1)
                    gs_embed2=torch.repeat_interleave(gs_embed,self.num_negative_samples,0)
                    pos_score =nn.Sigmoid()((gs_embed1 * pos_gs_embed).sum(-1))
                    neg_score =nn.Sigmoid()((gs_embed2 * neg_gs_embed).sum(-1))
                    gs_loss=nn.BCELoss()(pos_score,torch.ones(pos_score.shape[0]).cuda())+nn.BCELoss()(neg_score,torch.zeros(neg_score.shape[0]).cuda())
                    ae_loss=nn.BCELoss()(x_bar,x1)
                    if self.sample_num==2:
                        true_batch=self.FEAT[n_id[:batch_size],-1]
                        clas_loss=nn.BCELoss()(clas_out,true_batch.reshape(-1,1))
                        disc_loss=nn.BCELoss()(disc_out,true_batch.reshape(-1,1))
                    else:
                        true_batch=self.FEAT[n_id[:batch_size],-1].long()
                        clas_loss=nn.CrossEntropyLoss()(clas_out,true_batch)
                        disc_loss=nn.CrossEntropyLoss()(disc_out,true_batch)
                    loss=ae_loss*self.params.alpha1+gs_loss*self.params.alpha2+clas_loss*self.params.alpha3+disc_loss*self.params.alpha4
                    #####optimization#####
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    total_loss+=loss.item()
                    AE_loss+=ae_loss.item()
                    GS_loss+=gs_loss.item()
                    CLAS_loss+=clas_loss.item()
                    DISC_loss+=disc_loss.item()
                    aa+=1
                pbar.set_description('processed: %d' % (1 + epoch))
                pbar.set_postfix(total_loss=total_loss/aa,AE_loss=AE_loss/aa,GS_loss=GS_loss/aa,CLAS_loss=CLAS_loss/aa,DISC_loss=DISC_loss/aa)
                pbar.update(1)    
    def save_model(self):
        torch.save(self.model.state_dict(),self.params.model_file)
        print('Saving model to %s' % self.params.model_file)
        
    def load_model(self):
        saved_state_dict = torch.load(self.params.model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % self.params.model_file)
        
    def load_data(self,samples,feat_file,edge_file,meta_file,SEP=','):
        feat=pd.read_csv(feat_file[0],header=0,index_col=0)
        edge=np.loadtxt(edge_file[0],dtype=str)
        meta=pd.read_csv(meta_file[0],header=0,index_col=0)
        Batch={}
        for k in range(feat.shape[0]):
            Batch[k]=[0,feat.shape[0]-1]
        flags='_'+str(samples[0])
        for i in np.arange(1,len(samples)):
            n=feat.shape[0]
            feat=pd.concat((feat,pd.read_csv(feat_file[i],header=0,index_col=0)))
            edge=np.vstack((edge,np.loadtxt(edge_file[i],dtype=str)))
            meta=pd.concat((meta,pd.read_csv(meta_file[i],header=0,index_col=0)))
            for k in np.arange(n,feat.shape[0]):
                Batch[k]=[n,feat.shape[0]-1]
            flags=flags+'_'+str(samples[i])
            x=minmax_scale(feat.values,axis=1)
        feat=pd.DataFrame(x,index=feat.index,columns=feat.columns)
        meta=meta.loc[feat.index,:]
        xx=dict(zip(feat.index, range(feat.shape[0])))
        edge=pd.DataFrame(edge,columns=['Cell1','Cell2'])
        edge['Cell1']=edge['Cell1'].map(xx)
        edge['Cell2']=edge['Cell2'].map(xx)
        ub=np.unique(meta.loc[:,'batch'])
        Y=np.zeros((meta.shape[0],1))
        for i in range(len(ub)):
            Y[np.where(meta.loc[:,'batch']==ub[i])[0],0]=i
        xx=torch.FloatTensor(feat.values)
        yy=torch.LongTensor(Y)
        zz=torch.cat((xx,yy),dim=1)
        data = Data(edge_index=torch.LongTensor(np.array([edge['Cell1'].values.tolist(), edge['Cell2'].values.tolist()])), 
                    x=zz)
        adj=process_adj(edge.values,feat.shape[0])
        adj_list=adj.tolil().rows
        return data,adj_list,feat,meta,Batch,flags

