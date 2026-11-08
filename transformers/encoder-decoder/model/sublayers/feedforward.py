import torch.nn as nn

class FeedForwardNetwork(nn.Module):

    """
    This class implements a Feed Forward Neural Network module.

    Args:
        dm (int): Dimension of the model.
        dff (int): Dimension of the Feed Forward Network hidden layer.
        dropout (float, optional): Dropout rate. Default is 0.1.

    Attributes:
        w1 (torch.nn.Module): First linear transformation layer, with shape (dm, dff).
        w2 (torch.nn.Module): Second linear transformation layer, with shape (dff, dm).
        relu (torch.nn.Module): ReLU activation function.
        drop (torch.nn.Dropout): Dropout layer.
    """

    def __init__(self, dm, dff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dm, dff)
        self.w2 = nn.Linear(dff, dm)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        """
        Forward propagation of the Feed Forward Neural Network module.

        Args:
            x (torch.Tensor): Input tensor, with shape (batch_size, seq_len, dm).

        Returns:
            torch.Tensor: Output tensor, with shape (batch_size, seq_len, dm).
        """

        # inshape: x - (batch_size, seq_len, dm)
        
        # first linear transform with ReLU | shape: x - (batch_size, seq_len, dff)
        x = self.relu(self.w1(x))
        # drop neurons
        x = self.drop(x)
        # second linear transform | shape: out - (batch_size, seq_len, dm)
        out = self.w2(x)
        return out