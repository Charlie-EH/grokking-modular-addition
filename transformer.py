""" Recreation of Nanda et al. transformer (My own code)"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import einops

class Transformer(nn.Module):
    def __init__(self,d_vocab,d_model,n_ctx,d_head,num_heads,d_mlp):
        super().__init__()
        self.embed = nn.Embedding(d_vocab,d_model) # Associate a given integer input with a 128d vector
        self.pos_embed = nn.Embedding(n_ctx,d_model)    # associate a given position with a 128d vector
        self.attn = Attention(d_model,num_heads,d_head,n_ctx)
        self.mlp = nn.Sequential(nn.Linear(d_model,d_mlp),nn.ReLU(),nn.Linear(d_mlp,d_model))
        self.unembed = nn.Linear(d_model,d_vocab,bias=False)

    def forward(self,X):                                         # X is some set of tokens of shape (N,3), N is an, arbitrary number of tokens,
                                                                       # 3 is for a,b and = values 
        positions = torch.arange(X.shape[1],device=X.device)
        X = self.embed(X) +  self.pos_embed(positions)   # embed adds d_model (set in __init__) for (N,3,d_model=128) + (3,128) = (n,3,d_model)
                                                                       # now: X (N,position,d_model) (n,l,d)
        X = X + self.attn(X)
        X = X + self.mlp(X)
        X = self.unembed(X)
        return X[:,-1]

class Attention(nn.Module):
    def __init__(self,d_model,num_heads,d_head,n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads,d_head,d_model)/np.sqrt(d_model)) # gives h, i, d
        self.W_Q = nn.Parameter(torch.randn(num_heads,d_head,d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads,d_head,d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model,d_head*num_heads)/np.sqrt(d_model))
        self.register_buffer("mask",torch.tril(torch.ones((n_ctx,n_ctx))))
        self.d_head = d_head

    def forward(self,X):
        # n batch | h head | l query pos | m key pos | i d_head | d d_model | f joined heads
        k = torch.einsum('hid,nld->nhli',self.W_K,X) #'hid,nld->nhli' is \sigma_128 (num_heads,d_head,d_model)*(batch_size,positions,d_model) 
        q = torch.einsum('hid,nld->nhli',self.W_Q,X)
        v = torch.einsum('hid,nld->nhli',self.W_V,X)
        scores = torch.einsum("nhli,nhmi->nhlm", q,k)
        masked_scores = scores.masked_fill(self.mask == 0 , float("-inf"))
        pattern = torch.softmax(masked_scores/(self.d_head**0.5),dim=-1)
        z = torch.einsum("nhlm,nhmi->nhli",pattern,v) # l -> m for v : query becomes key?
        z_joined = einops.rearrange(z,'n h l i -> n l (h i)')
        out = torch.einsum('df,nlf->nld', self.W_O,z_joined)
         

        return out

if __name__ == "__main__":
    attn = Attention(d_model=128, num_heads=4, d_head=32, n_ctx=3)
    print(attn(torch.randn(5, 3, 128)).shape)            # expect torch.Size([5, 3, 128])

    model = Transformer(d_vocab=114, d_model=128, n_ctx=3, d_head=32, num_heads=4, d_mlp=512)
    tokens = torch.tensor([[3, 7, 113], [0, 112, 113]])
    print(model(tokens).shape)                           # expect torch.Size([2, 114])
    print(sum(p.numel() for p in model.parameters()))    # expect 226816