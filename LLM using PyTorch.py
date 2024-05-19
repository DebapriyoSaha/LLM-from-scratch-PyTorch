import torch
from torch import Tensor

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# In all the classes, look at the arguments passed to the __init__ function carefully
# Initialize the parameters with the seed values as mentioned in the assignments

class MHA(nn.Module):

  def __init__(self,dmodel,dq,dk,dv,heads):
    super(MHA,self).__init__()
  
    self.heads = heads
    self.dq = dq
    self.dk = dk
    self.dv = dv

    # Parameters initialization with specified seed values
    torch.manual_seed(43)
    self.WQ = nn.Parameter(torch.randn(dmodel, heads * dq))

    torch.manual_seed(44)
    self.WK = nn.Parameter(torch.randn(dmodel, heads * dk))

    torch.manual_seed(45)
    self.WV = nn.Parameter(torch.randn(dmodel, heads * dv))

    torch.manual_seed(46)
    self.WO = nn.Parameter(torch.randn(heads * dv, dmodel))

  # your method definitions go here (if you want to)

  def forward(self,H=None):
    '''
    Input: Size [BSxTxdmodel]
    Output: Size[BSxTxdmodel]
    '''
    # your code goes here

    if H is None:
        raise ValueError("Input tensor H must be provided.")

    # Get dimensions
    BS, T, dmodel = H.size()
    heads, dq, dk, dv = self.heads, self.dq, self.dk, self.dv

    # Linear projections
    Q = torch.matmul(H, self.WQ).view(BS, T, heads, dq).transpose(1,2)
    K = torch.matmul(H, self.WK).view(BS, T, heads, dk).transpose(1,2)
    V = torch.matmul(H, self.WV).view(BS, T, heads, dv).transpose(1,2)

    # Calculating Attention scores
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

    # Calculating Attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Calculating Weighted sum of values
    attention_output = torch.matmul(attention_weights, V)

    # Reshaping and concatenate heads
    attention_output = attention_output.transpose(1, 2).contiguous().view(BS, T, -1)

    # Calculating Linear projection
    linear_out = torch.matmul(attention_output, self.WO)

    return linear_out
  

class FFN(nn.Module):
  def __init__(self,dmodel,d_ff):
    super(FFN,self).__init__()
    torch.manual_seed(47)
    self.W1 = nn.Parameter(torch.randn(dmodel, d_ff))

    torch.manual_seed(48)
    self.W2 = nn.Parameter(torch.randn(d_ff, dmodel))

    self.activation = nn.ReLU()
  def forward(self,x):
    '''
    input: size [BSxTxdmodel]
    output: size [BSxTxdmodel]
    '''
    #your code goes here
    output = self.activation(torch.matmul(x, self.W1))  # [BSxTxd_ff]

    output = torch.matmul(output, self.W2)  # [BSxTxd_model]

    return output


class OutputLayer(nn.Module):

  def __init__(self,dmodel,vocab_size):
    super(OutputLayer,self).__init__()
    torch.manual_seed(49)
    # self.WL = nn.Linear(dmodel, vocab_size)
    self.WL = nn.Parameter(torch.randn(dmodel,vocab_size))

  def forward(self,representations):
    '''
    input: size [bsxTxdmodel]
    output: size [bsxTxvocab_size]
    Note: Do not apply the softmax. Just return the output of linear transformation
    '''
    # output = self.WL(representations)
    output = torch.matmul(representations, self.WL)
    return output

class MHCA(nn.Module):

  def __init__(self,dmodel,dq,dk,dv,heads):    
    super(MHCA,self).__init__()
    self.heads = heads
    self.dq = dq
    self.dk = dk
    self.dv = dv
    # your method definitions go here (if you want to)
    # Parameters initialization with specified seed values
    torch.manual_seed(43)
    self.WQ = nn.Parameter(torch.randn(dmodel, heads * dq))

    torch.manual_seed(44)
    self.WK = nn.Parameter(torch.randn(dmodel, heads * dk))

    torch.manual_seed(45)
    self.WV = nn.Parameter(torch.randn(dmodel, heads * dv))

    torch.manual_seed(46)
    self.WO = nn.Parameter(torch.randn(heads * dv, dmodel))

  # def forward(self,context=None,H=None):
  def forward(self,Enc_rep,Dec_rep):
    if Dec_rep is None:
        raise ValueError("Input tensor H must be provided.")

    BS, T1, dmodel = Dec_rep.size() #10x8x32
    _,T2,_ = Enc_rep.size()
    heads, dq, dk, dv = self.heads, self.dq, self.dk, self.dv

    # Linear projections
    Q = torch.matmul(Dec_rep, self.WQ).view(BS, T1, heads, dq).transpose(1,2)
    K = torch.matmul(Enc_rep, self.WK).view(BS, T2, heads, dk).transpose(1,2)
    V = torch.matmul(Enc_rep, self.WV).view(BS, T2, heads, dv).transpose(1,2)

    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

    # Attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    attn_output = torch.matmul(attn_weights, V)

    # Reshape and concatenate heads
    attn_output = attn_output.transpose(1, 2).contiguous().view(BS, T1, -1)

    # Linear projection
    out = torch.matmul(attn_output, self.WO)

    return out

class MHMA(nn.Module):

  def __init__(self,dmodel,dq,dk,dv,heads,seq_len,mask=None):
    '''
    seq_len: Helpful to create a causal mask if mask is None
    '''
    super(MHMA,self).__init__()   
    self.heads = heads
    self.dq = dq
    self.dk = dk
    self.dv = dv
    self.mask=mask

    # Parameters initialization with specified seed values
    torch.manual_seed(43)
    self.WQ = nn.Parameter(torch.randn(dmodel, heads * dq))

    torch.manual_seed(44)
    self.WK = nn.Parameter(torch.randn(dmodel, heads * dk))

    torch.manual_seed(45)
    self.WV = nn.Parameter(torch.randn(dmodel, heads * dv))

    torch.manual_seed(46)
    self.WO = nn.Parameter(torch.randn(heads * dv, dmodel))


  def forward(self,H=None):
    # implement forward method
    if H is None:
        raise ValueError("Input tensor H must be provided.")

    # Get dimensions
    BS, T, dmodel = H.size()
    heads, dq, dk, dv = self.heads, self.dq, self.dk, self.dv

    # Linear projections
    Q = torch.matmul(H, self.WQ).view(BS, T, heads, dq).transpose(1,2)
    K = torch.matmul(H, self.WK).view(BS, T, heads, dk).transpose(1,2)
    V = torch.matmul(H, self.WV).view(BS, T, heads, dv).transpose(1,2)

    # Calculating Attention scores
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

    #Applying Masking
    self.mask = torch.tril(torch.ones(T, T)).unsqueeze(0) == 0  # Lower triangle
    attention_scores = attention_scores.masked_fill(self.mask, float('-inf'))    
    
    # Calculating Attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Calculating Weighted sum of values
    attention_output = torch.matmul(attention_weights, V)

    # Reshaping and concatenate heads
    attention_output = attention_output.transpose(1, 2).contiguous().view(BS, T, -1)

    # Calculating Linear projection
    linear_out = torch.matmul(attention_output, self.WO)

    return linear_out

class PredictionHead(nn.Module):

  def __init__(self,dmodel,vocab_size):
    super(PredictionHead,self).__init__()
    torch.manual_seed(49)
    # self.WL = nn.Linear(dmodel, vocab_size)
    self.WL = nn.Parameter(torch.randn(dmodel,vocab_size))

  def forward(self,representations):
    '''
    input: size [bsxTxdmodel]
    output: size [bsxTxvocab_size]
    Note: Do not apply the softmax. Just return the output of linear transformation
    '''
    # output = self.WL(representations)
    output = torch.matmul(representations, self.WL)
    return output