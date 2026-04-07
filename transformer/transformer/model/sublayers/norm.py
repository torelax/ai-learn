import torch
import torch.nn as nn

class Norm(nn.Module):

    """
    This class implements a normalization layer.

    Args:
        dm (int): The dimension of the model.
        eps (float, optional): A small number to avoid division by zero. Default is 1e-6.

    Attributes:
        gamma (torch.nn.Parameter): A learnable scale parameter, with shape (dm).
        beta (torch.nn.Parameter): A learnable shift parameter, with shape (dm).
        eps (float): A small constant to ensure numerical stability. Default is 1e-6.
    """    

    def __init__(self, dm, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dm))
        self.beta = nn.Parameter(torch.zeros(dm))
        self.eps = eps

    def forward(self, x: torch.Tensor):

        """
        Forward propagation of the normalization layer.

        Args:
            x (torch.Tensor): Input tensor, with shape (batch_size, seq_len, dm).

        Returns:
            torch.Tensor: The normalized tensor, with shape (batch_size, seq_len, dm).
        """

        # inshape: x - (batch_size, seq_len, dm)

        # calc mean & variance (along dm)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=True, keepdim=True)
        # normalize, scale & shift | shape: out - (batch_size, seq_len, dm)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        out = norm * self.gamma + self.beta
        return out