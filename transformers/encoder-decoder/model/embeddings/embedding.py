import torch.nn as nn

class Embeddings(nn.Module):

    """
    This class represents an embedding layer for a transformer model.
    It is used to convert the input token ids into embeddings of a specified dimensionality.
    Inherits from the PyTorch nn.Module.

    Args:
        vocab_size (int): The size of the vocabulary.
        dm (int): The dimensionality of the embeddings.
        pad_id (int): The id that is used for padding in the input sequences.

    Attributes:
        embedding (torch.nn.Embedding): The embedding layer, with shape (vocab_size, dm).
    """

    def __init__(self, vocab_size, dm, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dm, padding_idx=pad_id)

    def forward(self, x):

        """
        This method applies the embedding layer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, seq_len).
        
        Returns:
            torch.Tensor: The ids converted to embeddings, with shape (batch_size, seq_len, dm).
        """

        # inshape: x - (batch_size, seq_len)

        # embed tokens to dm | shape: out - (batch_size, seq_len, dm)
        out = self.embedding(x)
        return out
    
    def unembedding(self):

        """
        This method retrieves the weight tensor from the embedding layer.
        
        Returns:
            torch.Tensor: The weight tensor of the embedding layer, with shape (vocab_size, dm).
        """

        # unembed tensor | shape: w - (vocab_size, dm)
        w = self.embedding.weight
        return w