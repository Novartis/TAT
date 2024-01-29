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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import TransformerEncoder

from skorch.regressor import NeuralNetRegressor


class TransformerMkII(nn.Module):
    def __init__(self, input_size, layer_size, output_size, hidden_size=512, n_modules = 1, n_heads = 4, drop = 0.1, max_seq_length = 230, create_output_network = True):
        super(TransformerMkII, self).__init__()
        self.n_modules = n_modules
        self.input_size = input_size
        self.layer_size = layer_size
        self.n_heads = n_heads
        self.create_output_network = create_output_network

        self.max_seq_length = max_seq_length # + 1 # + 1 because of token
        self.embed = nn.Linear(self.input_size, layer_size)                            
        self.pe = PositionalEncoding(d_model = layer_size, max_len = self.max_seq_length, dropout = 0) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=layer_size, nhead=n_heads, dropout = drop, dim_feedforward = layer_size)
        self.transformer_module = TransformerEncoder(encoder_layer, num_layers = self.n_modules)
                 
        if self.create_output_network:
            self.net = nn.Sequential(
                nn.Linear(layer_size, output_size)
            )

        self.init_weights()

    def init_weights(self):        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def get_embedding(self,x):
        n_batch_elements = x.shape[0]

        # embed onto feature space
        x = x.reshape(-1,self.input_size) #[bat*seq,emb]
        
        layer_input = self.embed(x)

        # reconfigure
        layer_input = layer_input.reshape(n_batch_elements,
                                          -1,
                                          self.layer_size) #[bat, seq, emb]
        

        layer_input = layer_input.permute(1,0,2)#[seq, bat, emb]


        # create positional embedings        
        layer_input = self.pe(layer_input) #still [seq, bat, emb]
        

        # prepend token
        CRM = False
        if CRM:
            layer_input = layer_input.permute(1,0,2) #[bat, seq, emb]
            flat = layer_input.view(n_batch_elements,-1) #[bat, seq*emb]
            tokens = torch.ones(n_batch_elements, self.layer_size).to(flat.device) # should we maybe do ones here? Cause maybe the zeros are zero-ing out 
            prepended = torch.cat([tokens, flat], dim = 1)
            layer_input = prepended.reshape(n_batch_elements,-1,self.layer_size) #[bat, seq,emb]


            # reformat to seq-bat-emb
            layer_input = layer_input.permute(1,0,2) #[seq, bat,emb]

        # apply transformer encoder
        layer_input = self.transformer_module(layer_input)
        

        # reshape to original shape
        layer_input = layer_input.permute(1,0,2) #[bat, seq,emb]


        # we could also return the last layer_input because that's the embedding actually
        if CRM:
            layer_input = layer_input[:,0,:] #, errr, this is selecting the first sequence in the batch?? No, it's the first element of the sequence
        else:
            layer_input = layer_input.mean(dim=1)
            
        return layer_input

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        if self.training:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros(logvar.shape).to(logvar.device)            
        return mu + eps*std


    def forward(self,x):
        layer_input = self.get_embedding(x)

        z = layer_input
        
        if self.create_output_network: #if this is an independent network
            out = self.net(z)

            if self.training:
                return out, z
            else:
                return out
        
        else: # if this is a part of a module
            return z

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x * math.sqrt(self.d_model)
        #x = x + self.pe[:x.size(0)]
        x = x + Variable(self.pe[:x.size(0)], \
        requires_grad=False).cuda()        
        #return self.dropout(x)
        return x







class TNetMkII(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        y_hat, z = y_pred
        mse = super().get_loss(y_hat, y_true, *args, **kwargs)
        loss = mse
        return loss
