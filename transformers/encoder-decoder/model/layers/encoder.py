import torch.nn as nn
from model.embeddings.embedding import Embeddings
from model.embeddings.pos_encoder import PositionalEncoder
from model.sublayers.attention import MultiHeadAttention
from model.sublayers.norm import Norm
from model.sublayers.feedforward import FeedForwardNetwork

class EncoderLayer(nn.Module):

    """
    This class represents a single layer of the Transformer's Encoder.

    Args:
        dm (int): Dimension of the model.
        dk (int): Dimension of the key.
        dv (int): Dimension of the value.
        nhead (int): Number of heads in the MultiHeadAttention module.
        dff (int): Dimension of the feed forward network model.
        bias (bool, optional): If set to False, the layers will not learn an additive bias. Default is False.
        dropout (float, optional): Dropout value. Default is 0.1.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-6.

    Attributes:
        multihead (model.sublayers.attention.MultiHeadAttention): MultiHeadAttention sublayer for contextual attention.
        feedforward (model.sublayers.feedforward.FeedForwardNetwork): FeedForwardNetwork sublayer.
        norm1, norm2 (model.sublayers.norm.Norm): Normalization sublayers.
        drop1, drop2 (torch.nn.Dropout): Dropout layers.
    """    

    def __init__(self, dm, dk, dv, nhead, dff, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.multihead = MultiHeadAttention(dm, dk, dv, nhead, bias=bias, dropout=dropout)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout=dropout)
        self.norm1 = Norm(dm, eps=eps)
        self.norm2 = Norm(dm, eps=eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        """
        Forward propagation of the Encoder Layer.

        Args:
            src (torch.Tensor): Source sequences, with shape (batch_size, src_len, dm).
            src_mask (torch.Tensor, optional): Tensor representing the source mask, with shape (batch_size, 1, src_len). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor, with shape (batch_size, src_len, dm) and attention weights, with shape (batch_size, nhead, q_len, k_len).
        """

        # inshape: src - (batch_size, src_len, dm)

        # get context | shape - x_out (batch_size, src_len, dm)
        x = src
        x_out, attn = self.multihead(x, x, x, mask=src_mask)
        # drop neurons
        x_out = self.drop1(x_out)
        # add & norm (residual connections) | shape: x - (batch_size, src_len, dm)
        x = self.norm1(x + x_out)

        # linear transforms | shape: x_out (batch_size, src_len, dm)
        x_out = self.feedforward(x) 
        # drop neurons
        x_out = self.drop2(x_out)
        # add & norm (residual connections) | shape: out - (batch_size, src_len, dm)
        out = self.norm2(x + x_out)
        return out, attn

class Encoder(nn.Module):

    """
    This class represents the Encoder module of the Transformer.

    Args:
        vocab_size (int): Size of the vocabulary.
        maxlen (int): Maximum length of the input sequence.
        pad_id (int): The id used to pad sequences.
        dm (int): Dimension of the model.
        dk (int): Dimension of the key.
        dv (int): Dimension of the value.
        nhead (int): Number of heads in the MultiHeadAttention modules.
        dff (int): Dimension of the feed forward network model.
        layers (int, optional): Number of encoder blocks in the encoder stack. Default is 6.
        bias (bool, optional): If set to False, the layers will not learn an additive bias. Default is False.
        dropout (float, optional): Dropout value. Default is 0.1.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-6.
        scale (bool, optional): Whether to scale the output of the positional encoder. Default is True.
    
    Attributes:
        embeddings (model.embeddings.embedding.Embeddings): Embedding layer for the source sequences, with shape (vocab_size, dm).
        pos_encodings (model.embeddings.pos_encoder.PositionalEncoder): Positional Encoder layer, with shape (1, maxlen, dm).
        stack (nn.ModuleList): Stack of EncoderLayer modules.
    """

    def __init__(self, vocab_size, maxlen, pad_id, dm, dk, dv, nhead, dff, layers=6, bias=False, 
                 dropout=0.1, eps=1e-6, scale=True) -> None:
        super().__init__()
        self.embeddings = Embeddings(vocab_size, dm, pad_id)
        self.pos_encodings = PositionalEncoder(dm, maxlen, dropout=dropout, scale=scale)
        self.stack = nn.ModuleList([EncoderLayer(dm, dk, dv, nhead, dff, bias=bias, dropout=dropout, eps=eps) 
                                    for l in range(layers)])

    def forward(self, src, src_mask=None):

        """
        Forward propagation of the Encoder module.

        Args:
            src (torch.Tensor): Source sequences, with shape (batch_size, src_len).
            src_mask (torch.Tensor, optional): Tensor representing the source mask, with shape (batch_size, 1, src_len). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor, with shape (batch_size, src_len, dm) and attention weights, with shape (batch_size, nhead, q_len, k_len)
        """

        # inshape: src - (batch_size, src_len, dm) src_mask - (batch_size, 1, src_len)

        # embeddings + positional encodings | shape: x - (batch_size, src_len, dm)
        x = self.embeddings(src)
        x = self.pos_encodings(x)
        # pass src through stack of encoders (out of layer is in for next)
        for encoder in self.stack:
            x, attn = encoder(x, src_mask=src_mask)
        # shape: out - (batch_size, src_len, dm)
        out = x
        return out, attn