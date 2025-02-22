import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d_utils

from math import sqrt

class PosEncoding(nn.Module):
    def __init__(self,embedding_dim,max_len,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe_map = torch.zeros((max_len,embedding_dim))

        pos = torch.arange(0,max_len).to(torch.float32).unsqueeze(1)
        
        embedding_index = torch.arange(0,embedding_dim).to(torch.float32)*2

        self.pe_map[:,::2] = torch.sin(pos/(torch.tensor(10000)**(embedding_index[::2]/embedding_dim)))
        self.pe_map[:,1::2] = torch.cos(pos*(torch.tensor(10000)**(embedding_index[1::2]/embedding_dim)))
    
    def forward(self,word_embedding,batch=False):
        return self.dropout(word_embedding + self.pe_map[:word_embedding.shape[batch+0]])

class AttentionUnit(nn.Module):
    def __init__(self,embedding_dim,mask=None,batch=False):
        super().__init__()
        self.mask = mask
        self.embedding_dim = embedding_dim
        self.q = nn.Linear(embedding_dim,embedding_dim)
        self.v = nn.Linear(embedding_dim,embedding_dim)
        self.k = nn.Linear(embedding_dim,embedding_dim)
        self.batch = batch
        self.row_dim = 0+batch
        self.col_dim = 1+batch
        self.scale = sqrt(embedding_dim)

    def get_qvk(self,q,v,k):
        q = self.q(q)
        v = self.v(v)
        k = self.k(k)
        return q,v,k

    def forward(self,q,v,k):
        q,v,k = self.get_qvk(q,v,k)

        sims = torch.matmul(q,k.transpose(self.row_dim,self.col_dim))/self.scale

        if self.mask:
            if self.batch:
                mask = torch.ones((q.shape[0],q.shape[1],q.shape[1]))
            else:
                mask = torch.ones((q.shape[0],q.shape[0]))
            
            mask = torch.triu(mask,1)
            mask = mask == 1

            sims.masked_fill_(mask,-1e9)
        
        sims = F.softmax(sims,self.col_dim)

        return torch.matmul(sims,v)

class MultiheadAttention(nn.Module):
    def __init__(self,embedding_dim,n_head,mask=None,batch=False):
        super().__init__()
        self.chunk_size = embedding_dim//n_head
        self.n_chunk = embedding_dim//self.chunk_size
        self.remain = embedding_dim%n_head
        self.heads = nn.ModuleList([AttentionUnit(self.chunk_size,mask,batch) for i in range(self.n_chunk)])
        if self.remain:
            self.heads.append(AttentionUnit(self.remain,mask,batch))
    
    def forward(self,x):
        x = [head(chunk,chunk,chunk) for chunk,head in zip(torch.split(x,self.chunk_size,dim=-1),self.heads)]
        x = torch.cat(x,-1)
        return x

class MLP(nn.Module):
    def __init__(self,embedding_dim,dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(embedding_dim,embedding_dim*4),
                                   nn.ReLU(),
                                   nn.Linear(embedding_dim*4,embedding_dim),
                                   nn.Dropout(dropout))
    
    def forward(self,x):
        x = self.layer(x)
        return x

class Block(nn.Module):
    def __init__(self,embedding_dim,n_head,dropout,mask=None,batch=False):
        super().__init__()
        self.attention = MultiheadAttention(embedding_dim,n_head,mask,batch)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim,dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self,embedding_dim,max_len,vocab_size,n_head,dropout,n_block,batch=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.pos_encoding = PosEncoding(embedding_dim,max_len,dropout)
        self.blocks = nn.Sequential(*[Block(embedding_dim,n_head,dropout,True,batch) for i in range(n_block)])
        self.ln = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim,vocab_size)
        self.batch = batch

    def forward(self,x):
        x = self.embedding(x)
        pos_encoded = self.pos_encoding(x,self.batch)
        attention_encoded = self.blocks(pos_encoded)
        x = pos_encoded + attention_encoded
        x = self.ln(x)
        x = self.fc(x)
        return x
