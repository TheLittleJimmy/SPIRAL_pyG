#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
from torch_geometric.nn import SAGEConv

from pytorch_revgrad import RevGrad


# In[ ]:


class NeighborSampler(RawNeighborSampler):
    def __init__(self,edge_index, sizes,batch_dict=None,adj_lists=None,node_idx=None,num_nodes=None,return_e_id=False,transform=None,P=1.0,Q=1.0,walks_per_node=6.0,walk_length=1.0,negative_walk_length=6.0,num_negative_samples=6.0,**kwargs):
        super().__init__(edge_index, sizes,node_idx,num_nodes,return_e_id,transform,**kwargs)
        self.batch_dict=batch_dict
        self.adj_lists=adj_lists
        self.P=P
        self.Q=Q
        self.walks_per_node=walks_per_node
        self.walk_length=walk_length
        self.negative_walk_length=negative_walk_length
        self.num_negative_samples=num_negative_samples
    def sample(self, batch):
        batch = torch.tensor(batch)
        batch1 = batch.repeat(self.walks_per_node)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch1, walk_length=self.walk_length,p=self.P,q=self.Q,coalesced=False)[:, 1]
        
        neg_batch = self.get_negtive_nodes(batch.numpy())

        batch = torch.cat([batch, pos_batch, torch.tensor(neg_batch[:,1])], dim=0)
        return super().sample(batch)
    
    def get_negtive_nodes(self,nodes):
        negtive_pairs=[]
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.negative_walk_length):
                current = set()
                for outer in frontier:
                    current |= set(self.adj_lists[int(outer)])
                frontier = current - neighbors
                neighbors |= current
            a=np.arange(self.batch_dict[node][0],self.batch_dict[node][1])
            train_nodes = set(list(a))
            far_nodes = train_nodes - neighbors
            neg_samples = random.choices(list(far_nodes),k=self.num_negative_samples)
#             neg_samples = random.sample(far_nodes, num_negative_samples)
            negtive_pairs.extend([[node, neg_node] for neg_node in neg_samples])
        return np.array(negtive_pairs)


# In[ ]:


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0,bias=True):
    net = nn.Sequential()
    for i in range(1, len(layers)):
        net1=[]
        net1.append(nn.Linear(layers[i-1], layers[i],bias=bias))
        if bn:
            net1.append(nn.BatchNorm1d(layers[i]))
        if activation is not None:
            net1.append(activation)
        if dropout > 0:
            net1.append(nn.Dropout(dropout))
        net1=nn.Sequential(*net1)
        net.add_module('layer'+str(i),net1)
    return net

class Encoder(nn.Module):
    def __init__(self,dims):
        super(Encoder, self).__init__()
        [x_dim,h_dim1,z_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([x_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=x_dim
        self.z_layer=nn.Linear(h_dim1, z_dim)
    def forward(self, x):
        en_h=[]
        a=x
        if self.n_hdim>0:
            for i in np.arange(self.n_hdim):
                a=self.hidden1[i](a)
                en_h.append(a)
        if len(en_h)>0:
            z=self.z_layer(en_h[-1])
        else:
            z=self.z_layer(x)
        return en_h,z
    
class Decoder(nn.Module):
    def __init__(self,dimsR):
        super(Decoder, self).__init__()
        [z_dim,h_dim1,x_dim]=dimsR
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.x_layer = nn.Linear(h_dim1, x_dim)
    def forward(self, z,act=nn.Sigmoid()):
        de_h=[]
        a=z
        if self.n_hdim>0:
            for i in np.arange(self.n_hdim):
                a=self.hidden1[i](a)
                de_h.append(a)
        if len(de_h)>0:
            x_bar=self.x_layer(de_h[-1])
        else:
            x_bar=self.x_layer(z)
        if act is not None:
            return de_h,act(x_bar)
        else:
            return de_h,x_bar
        
class AE(nn.Module):
    def __init__(self, dims,dimsR):
        super(AE, self).__init__()
        self.en=Encoder(dims)
        self.de=Decoder(dimsR)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x,de_act=None):
        enc_h,z=self.en(x)
        dec_h,x_bar=self.de(z,de_act)
        return enc_h,z,dec_h,x_bar
    
class Classifier(nn.Module):
    def __init__(self,dims):
        super(Classifier, self).__init__()
        [z_dim,h_dim1,out_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.out=nn.Linear(h_dim1, out_dim)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, z,act):
        h=self.hidden1(z)
        out=self.out(h)
        if act is not None:
            return act(out)
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, dims):
        super(Discriminator, self).__init__()
        [z_dim,h_dim1,out_dim]=dims
        self.n_hdim=len(h_dim1)
        if self.n_hdim>0:
            self.hidden1=build_mlp([z_dim]+h_dim1,activation=nn.ReLU(), bn=True, dropout=0,bias=True)
            h_dim1=h_dim1[-1]
        else:
            self.hidden1=nn.Identity()
            h_dim1=z_dim
        self.out=nn.Linear(h_dim1, out_dim)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, z,act):
        h=self.hidden1(z)
        out=self.out(h)
        if act is not None:
            return act(out)
        else:
            return out

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,agg_class):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.bias = []
        for i in range(num_layers-1):
            in_channels = in_channels if i == 0 else hidden_channels[i-1]
            self.convs.append(SAGEConv(in_channels, hidden_channels[i],aggr=agg_class,bias=True,root_weight=False))
            self.lins.append(nn.Linear(in_channels, hidden_channels[i], bias=True))
#             self.bias.append(nn.Parameter(torch.FloatTensor(hidden_channels[i]),requires_grad=True).cuda())
        self.init_weights()
            
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
#                 torch.nn.init.xavier_normal_(m.weight.data)               
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
#                 self.default_init(m)
            if isinstance(m, SAGEConv):
                torch.nn.init.xavier_uniform_(m.lin_l.weight.data)
#                 torch.nn.init.xavier_normal_(m.lin_l.weight.data)
                if m.lin_l.bias is not None:
                    m.lin_l.bias.data.fill_(0.0)
#                 self.default_init(m.lin_l)
                
    def default_init(self,m):            
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)


    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            edge_index=edge_index.cuda()
            h = self.convs[i]((x, x_target), edge_index)
            x_target = self.lins[i](x_target)
            x = x_target+h
#             x = x+self.bias[i]
            if i != self.num_layers - 2:
                x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            x = self.lins[i](x)+h
            if i != self.num_layers - 2:
                x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
        return x

class A_G_Combination(nn.Module):
    def __init__(self, AEdims, AEdimsR,GSdims,agg_class,beta):
        super(A_G_Combination, self).__init__()
        self.ae=AE(AEdims, AEdimsR)
        self.gs=SAGE(AEdims[0],GSdims,len(GSdims)+1,agg_class)
        self.beta=beta
    def forward(self,x,adjs,de_act):
        x1=x[:adjs[-1].size[1]]
        _,ae_z=self.ae.en(x1)
        gs_z=self.gs(x,adjs)
#         z=self.combine_layer(torch.cat((ae_z,gs_z),dim=1))
        z=(1-self.beta)*ae_z+self.beta*gs_z
        _,x_bar=self.ae.de(z,de_act)
        final_z=[ae_z,gs_z,z]
        return final_z,x_bar,x1
    def full_forward(self,x1,edge_index,de_act):
        _,ae_z=self.ae.en(x1)
        gs_z=self.gs.full_forward(x1,edge_index)
        z=(1-self.beta)*ae_z+self.beta*gs_z
        _,x_bar=self.ae.de(z,de_act)
        final_z=[ae_z,gs_z,z]
        return final_z,x_bar,x1
    
class A_G_Combination_DA_complex(nn.Module):
    def __init__(self, AEdims, AEdimsR,GSdims,agg_class,beta,znoise_dim,CLdims,DIdims):
        super(A_G_Combination_DA_complex, self).__init__()
        self.znoise_dim=znoise_dim
        self.agc=A_G_Combination(AEdims, AEdimsR,GSdims,agg_class,beta)
        self.clas=Classifier(CLdims)
        self.disc=Discriminator(DIdims)
    def forward(self,x,adjs,lamda,de_act,cl_act):
        self.revgrad=RevGrad(lamda)
        final_z,x_bar,x1=self.agc(x,adjs,de_act)
        ae_z,gs_z,z=final_z
        znoise=z[:,:self.znoise_dim]
        zbio=z[:,self.znoise_dim:]
        clas_out=self.clas(znoise,act=cl_act)
        disc_out=self.disc(self.revgrad(zbio),act=cl_act)
        ae_out=[x_bar,x1]
        return final_z,ae_out,clas_out,disc_out
    def full_forward(self,x1,edge_index,lamda,de_act,cl_act):
        final_z,x_bar,x1=self.agc.full_forward(x1,edge_index,de_act)
        ae_out=[x_bar,x1]
        return final_z,ae_out

