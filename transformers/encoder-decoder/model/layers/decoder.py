import torch.nn as nn
from model.embeddings.embedding import Embeddings
from model.embeddings.pos_encoder import PositionalEncoder
from model.sublayers.attention import MultiHeadAttention
from model.sublayers.norm import Norm
from model.sublayers.feedforward import FeedForwardNetwork

class DecoderLayer(nn.Module):

    """
    This class represents a single layer of the Transformer's Decoder.

    Args:
        dm (int): Dimension of the model.
        dk (int): Dimension of the key.
        dv (int): Dimension of the value.
        nhead (int): Number of heads in the masked and unmasked MultiHeadAttention module.
        dff (int): Dimension of the feed forward network model.
        bias (bool, optional): If set to False, the layers will not learn an additive bias. Defaults to False.
        dropout (float, optional): Dropout value. Defaults to 0.1.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-6.

    Attributes:
        maskmultihead (model.sublayers.attention.MultiHeadAttention): MultiHeadAttention sublayer for masked attention.
        multihead (model.sublayers.attention.MultiHeadAttention): MultiHeadAttention sublayer for non-masked attention.
        feedforward (model.sublayers.feedforward.FeedForwardNetwork): FeedForwardNetwork sublayer.
        norm1, norm2, norm3 (model.sublayers.norm.Norm): Normalization sublayers.
        drop1, drop2, drop3 (nn.Dropout): Dropout layers.
    """    

    def __init__(self, dm, dk, dv, nhead, dff, bias=False, dropout=0.1, eps=1e-6) -> None:
        super().__init__()
        self.maskmultihead = MultiHeadAttention(dm, dk, dv, nhead, bias=bias, dropout=dropout)
        self.multihead = MultiHeadAttention(dm, dk, dv, nhead, bias=bias, dropout=dropout)
        self.feedforward = FeedForwardNetwork(dm, dff, dropout=dropout)
        self.norm1 = Norm(dm, eps=eps)
        self.norm2 = Norm(dm, eps=eps)
        self.norm3 = Norm(dm, eps=eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, src, trg, src_mask=None, trg_mask=None):

        """
        Forward propagation of the Decoder Layer.

        Args:
            src (torch.Tensor): Source sequences, with shape (batch_size, src_len, dm).
            trg (torch.Tensor): Target sequences, with shape (batch_size, trg_len, dm).
            src_mask (torch.Tensor, optional): Tensor representing the source mask, with shape (batch_size, 1, src_len). Default is None.
            trg_mask (torch.Tensor, optional): Tensor representing the target mask, with shape (batch_size, trg_len, trg_len). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor, with shape (batch_size, trg_len, dm); both masked and unmasked attention weights, with shape (batch_size, nhead, q_len, k_len)
        """

        # inshape: src - (batch_size src_len, dm) trg - (batch_size, trg_len, dm) \
        # src_mask - (batch_size, 1 src_len) trg_mask - (batch_size trg_len, trg_len)/(batch_size, 1 , trg_len)

        # calc masked context | shape: x_out - (batch_size, trg_len, dm)
        x = trg
        x_out, attn1 = self.maskmultihead(x, x, x, mask=trg_mask)
        # drop neurons
        x_out = self.drop1(x_out)
        # add & norm (residual connections) | shape: x - (batch_size, trg_len, dm)
        x = self.norm1(x + x_out)

        # calc context | shape: x_out - (batch_size, trg_len, dm)
        x_out, attn2 = self.multihead(x, src, src, mask=src_mask)
        # drop neurons
        x_out = self.drop2(x_out)
        # add & norm (residual connections) | shape: x - (batch_size, trg_len, dm)
        x = self.norm2(x + x_out)

        # calc linear transforms | shape: x_out - (batch_size, trg_len, dm)
        x_out = self.feedforward(x)
        # drop neurons
        x_out = self.drop3(x_out)
        # add & norm (residual connections) | shape: out - (batch_size, trg_len, dm)
        out = self.norm3(x + x_out)
        return out, attn1, attn2
    
class Decoder(nn.Module):

    """
    This class represents the Decoder module of the Transformer.

    Args:
        vocab_size (int): Size of the vocabulary.
        maxlen (int): Maximum length of the input sequence.
        pad_id (int): The ID used to pad sequences.
        dm (int): Dimension of the model.
        dk (int): Dimension of the key.
        dv (int): Dimension of the value.
        nhead (int): Number of heads in the maskefd and unmasked MultiHeadAttention modules.
        dff (int): Dimension of the feed forward network model.
        layers (int, optional): Number of decoder blocks in the decoder stack. Defaults to 6.
        bias (bool, optional): If set to False, the layers will not learn an additive bias. Defaults to False.
        dropout (float, optional): Dropout value. Defaults to 0.1.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-6.
        scale (bool, optional): Whether to scale the output of the positional encoder. Defaults to True.

    Attributes:
        embeddings (model.embeddings.embedding.Embeddings): Embedding layer for the target sequences, with shape (vocab_size, dm).
        pos_encodings (model.embeddings.pos_encoder.PositionalEncoder): Positional Encoder layer, with shape (1, maxlen, dm).
        stack (nn.ModuleList): Stack of DecoderLayer modules.
    """    

    def __init__(self, vocab_size, maxlen, pad_id, dm, dk, dv, nhead, dff, layers=6, bias=False, 
                 dropout=0.1, eps=1e-6, scale=True) -> None:
        super().__init__()
        self.embeddings = Embeddings(vocab_size, dm, pad_id)
        self.pos_encodings = PositionalEncoder(dm, maxlen, dropout=dropout, scale=scale)
        self.stack = nn.ModuleList([DecoderLayer(dm, dk, dv, nhead, dff, bias=bias, dropout=dropout, eps=eps) 
                                    for l in range(layers)])
        
    def forward(self, src, trg, src_mask=None, trg_mask=None):

        """
        Forward propagation of the Decoder module.

        Args:
            src (torch.Tensor): Source sequences, with shape (batch_size, src_len, dm).
            trg (torch.Tensor): Target sequences, with shape (batch_size, trg_len, dm).
            src_mask (torch.Tensor, optional): Tensor representing the source mask, with shape (batch_size, 1, src_len). Default is None.
            trg_mask (torch.Tensor, optional): Tensor representing the target mask, with shape (batch_size, trg_len, trg_len). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor, with shape (batch_size, trg_len, dm); both masked and unmasked attention weights, with shape (batch_size, nhead, q_len, k_len)
        """

        # inshape: src - (batch_size, src_len, dm) trg - (batch_size, trg_len, dm)

        # embeddings + positional encodings | shape: x - (batch_size, trg_len, dm)
        x = self.embeddings(trg)
        x = self.pos_encodings(x)

        # pass src & trg through stack of decoders (out of layer is in for next)
        for decoder in self.stack:
            x, attn1, attn2 = decoder(src, x, src_mask=src_mask, trg_mask=trg_mask)
        out = x
        return out, attn1, attn2