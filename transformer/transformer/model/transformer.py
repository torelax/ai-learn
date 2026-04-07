import torch.nn as nn
import torch.nn.init as init
from model.layers.encoder import Encoder
from model.layers.decoder import Decoder

class Transformer(nn.Module):

    """
    This class represents the main Transformer architecture.

    Args:
        vocab_enc (int): Vocabulary size of the encoder.
        vocab_dec (int): Vocabulary size of the decoder.
        maxlen (int): Maximum length of the input sequence.
        pad_id (int): The ID used to pad sequences.
        dm (int, optional): Dimension of the model. Default is 512.
        dk (int, optional): Dimension of the key. Default is 64.
        dv (int, optional): Dimension of the value. Default is 64.
        nhead (int, optional): Number of heads in the MultiHeadAttention modules. Default is 8.
        layers (int, optional): Number of encoder blocks in the encoder stack and decoder blocks in the decoder stack. Default is 6.
        dff (int, optional): Dimension of the feed forward network model. Default is 2048.
        bias (bool, optional): If set to False, the layers will not learn an additive bias. Default is False.
        dropout (float, optional): Dropout value. Default is 0.1.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-6.
        scale (bool, optional): Whether to scale the output of the positional encoder. Default is True.

    Attributes:
        encoder (model.layers.encoder.Encoder): Encoder module.
        decoder (model.layers.decoder.Decoder): Decoder module.
        linear (torch.nn.Linear): Linear layer to transform decoder output, with shape (dm, vocab_dec).
    """    
    
    def __init__(self, vocab_enc, vocab_dec, maxlen, pad_id, dm=512, dk=64, dv=64, nhead=8, layers=6, 
                dff=2048, bias=False, dropout=0.1, eps=1e-6, scale=True) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_enc, maxlen, pad_id, dm, dk, dv, nhead, dff, 
                        layers=layers, bias=bias, dropout=dropout, eps=eps, scale=scale)          
        self.decoder = Decoder(vocab_dec, maxlen, pad_id, dm, dk, dv, nhead, dff, 
                        layers=layers, bias=bias, dropout=dropout, eps=eps, scale=scale)
        self.linear = nn.Linear(dm, vocab_dec)
        self.maxlen = maxlen
        self.pad_id = pad_id
        self.apply(xavier_init)

    def forward(self, src, trg, src_mask=None, trg_mask=None):

        """
        Forward propagation of the Transformer module.

        Args:
            src (torch.Tensor): Source sequences, with shape (batch_size, src_len).
            trg (torch.Tensor): Target sequences, with shape (batch_size, trg_len).
            src_mask (torch.Tensor, optional): Tensor representing the source mask, with shape (batch_size, 1, src_len). Default is None.
            trg_mask (torch.Tensor, optional): Tensor representing the target mask, with shape (batch_size, trg_len, trg_len). Default is None.

        Returns:
            torch.Tensor: Output tensor, with shape (batch_size, trg_len, vocab_size).
        """

        # inshape: src - (batch_size, src_len) trg - (batch_size, trg_len)\
        # src_mask - (batch_size, 1, src_len) trg_mask - (batch_size, trg_len, trg_len)
        
        # encode embeddings | shape: e_out - (batch_size, src_len, dm)
        e_out, attn = self.encoder(src, src_mask=src_mask)

        # decode embeddings | shape: d_out - (batch_size, trg_len, dm)
        d_out, attn, attn = self.decoder(e_out, trg, src_mask=src_mask, trg_mask=trg_mask)
        # linear project decoder output | shape: out - (batch_size, trg_len, vocab_size)
        out = self.linear(d_out)
        return out

def xavier_init(module):

    """
    This function applies Xavier uniform initialization to the weight of the given module.
    
    Args:
        module (torch.nn.Module): The module which weights are to be initialized.
    """

    if hasattr(module, "weight") and module.weight.dim() > 1:
        init.xavier_uniform_(module.weight.data)