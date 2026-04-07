import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):

    """
    This class implements the scaled dot product attention mechanism.

    Args:
        dk (int): Dimension of the key vectors.
        dropout (float, optional): Dropout rate. Default is None.

    Attributes:
        dk (int): Dimension of the key vectors.
        drop (torch.nn.Dropout): Dropout layer.
        softmax (torch.nn.Softmax): Softmax layer for attention calculation.
    """

    def __init__(self, dk, dropout=None) -> None:
        super().__init__()
        self.dk = dk
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None):

        """
        Forward propagation of scaled dot product attention mechanism.

        Args:
            q (torch.Tensor): Query vectors, with shape (batch_size, *n_head, q_len, dk).
            k (torch.Tensor): Key vectors, with shape (batch_size, *n_head, k_len, dk).
            v (torch.Tensor): Value vectors, with shape (batch_size, *n_head, k_len, dv).
            mask (torch.Tensor, optional): Mask to prevent attention at certain positions. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The context vector, with shape (batch_size, *n_head, q_len, dv) and the attention weights, with shape (batch_size, *n_head, q_len, k_len).
        """

        # inputs are projected | shape: q - (batch_size, *n_head, q_len, dk) k - (batch_size, *n_head, k_len, dk)  v - (batch_size, *n_head, k_len, dv)

        # compute dot prod w/ q & k then normalize | shape: similarities - (batch_size, *n_head, q_len, k_len)
        similarities = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dk)

        # apply mask (if required)
        if mask is not None:
            mask = mask.unsqueeze(1) # for multi-head attention
            similarities = similarities.masked_fill(mask == 0,-1e9)

        # compute attention weights | shape: attention - (batch_size, *n_head, q_len, k_len)
        attention = self.softmax(similarities)
        # drop attention weights
        attention = self.drop(attention)

        # compute context given v | shape: context - (batch_size, *n_head, q_len, dv)
        context = torch.matmul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    """
    This class implements the multi-head attention mechanism.

    Args:
        dm (int): Dimension of the model.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        nhead (int): Number of attention heads.
        bias (bool, optional): Whether to include bias in linear layer weights. Default is False.
        dropout (float, optional): Dropout rate. Default is None.

    Attributes:
        dm (int): Dimension of the model.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        nhead (int): Number of attention heads.
        wq (torch.nn.Module): Linear layer for query, with shape (dm, dk * nhead)
        wk (torch.nn.Module): Linear layer for key, with shape (dm, dk * nhead)
        wv (torch.nn.Module): Linear layer for value, with shape (dm, dv * nhead)
        wo (torch.nn.Module): Final linear layer, with shape (dv * nhead, dm)
        scaled_dot_prod_attn (model.sublayers.attention.ScaledDotProductAttention): ScaledDotProductAttention module.
    """

    def __init__(self, dm, dk, dv, nhead, bias=False, dropout=None):
        super().__init__()
        if dm % nhead != 0:
            raise ValueError("Embedding dimensions (dm) must be divisble by number of heads (nhead)")
        self.dm = dm
        self.dk = dk
        self.dv = dv
        self.nhead = nhead
        self.wq = nn.Linear(dm, dk * nhead, bias=bias)
        self.wk = nn.Linear(dm, dk * nhead, bias=bias)
        self.wv = nn.Linear(dm, dv * nhead, bias=bias)
        self.wo = nn.Linear(dv * nhead, dm)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(dk, dropout=dropout)

    def forward(self, q, k, v, mask=None):

        """
        Forward propagation of multi-head attention mechanism.

        Args:
            q (torch.Tensor): Query vectors, with shape (batch_size, q_len, dm).
            k (torch.Tensor): Key vectors, with shape (batch_size, k_len, dm).
            v (torch.Tensor): Value vectors, with shape (batch_size, k_len, dm).
            mask (torch.Tensor, optional): Mask to prevent attention at certain positions. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The context vector, with shape (batch_size, q_len, dm) and the attention weights, with shape (batch_size, nhead, q_len, k_len).
        """

        # inshape: q - (batch_size, q_len, dm) k & v - (batch_size, k_len, dm)
        batch_size, q_len, k_len = q.size(0), q.size(1), k.size(1)

        # linear projections into heads | shape: q - (batch_size, nhead, q_len, dk) k - (batch_size, nhead, k_len, dk) v - (batch_size, nhead, k_len, dv)
        q = self.wq(q).view(batch_size, q_len, self.nhead, self.dk).transpose(1, 2)
        k = self.wk(k).view(batch_size, k_len, self.nhead, self.dk).transpose(1, 2)
        v = self.wv(v).view(batch_size, k_len, self.nhead, self.dv).transpose(1, 2)

        # get context & attn weights | shape: attention - (batch_size, nhead, q_len, k_len) context - (batch_size, nhead, q_len, dv)
        context, attention = self.scaled_dot_prod_attn(q, k, v, mask=mask)

        # concat heads | shape: context - (batch_size, q_len, dm)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.dm)

        # project context vector | shape: context - (batch_size, q_len, dm)
        context = self.wo(context)
        return context, attention