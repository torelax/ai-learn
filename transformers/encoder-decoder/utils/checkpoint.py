import torch 
from utils.functional import create_path

class Checkpoint:

    """
    A class used for storing and loading the dynamic states of modules and metrics
    during the training of the Transformer.

    Args:
        model (model.transformer.Transformer): The transformer network. 
        optimizer (torch.optim.Optimizer, optional): The optimizer object assigned to the transformer network. Default is None.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau, optional): The scheduler assigned to the optimizer. Default is None.
        frequency (int, optional): The frequency at which the checkpoint will make a save. Default is None.
        path (str, optional): The path to save the checkpoint. Default is None.
        overwrite (bool, optional): If True, overwrite the existing checkpoint at the same path. Default is False.

    Attributes:
        model (model.transformer.Transformer): The transformer network. 
        optimizer (torch.optim.Optimizer): The optimizer object assigned to the transformer network.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The scheduler assigned to the optimizer.
        frequency (int): The frequency at which the checkpoint will make a save.
        path (str): The path to save the checkpoint.
        overwrite (bool): If True, overwrite the existing checkpoint at the same path.
        epoch (int): The current epoch count.
        dict (dict): A dictionary to store additional states.
    """

    def __init__(self, model, optimizer=None, scheduler=None, frequency=None, path=None, overwrite=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.frequency = frequency
        self.path = path
        self.overwrite = overwrite
        self.epoch = 0
        self.dict = dict()
        
    def check(self, **kwargs):

        """
        Saves the current state if the current epoch is a multiple of the frequency.

        Returns:
            bool: True if a checkpoint is saved, False otherwise.
        """

        if self.frequency is None:
            self.frequency = 1
        # update metric states
        self.epoch += 1
        # save current state
        if self.epoch % self.frequency == 0:
            self.save(**kwargs)
            return True
        return False

    def save(self, **kwargs):

        """
        Saves the current state of the model, optimizer, and scheduler.
        Additional states can be saved by passing them as keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments for additional states to save.
        """

        # create path (if non-existent)
        if self.path is None:
            self.path = "checkpoint.pt"
        create_path(self.path)
        path = f"{self.path[:-3]}-{self.epoch}{self.path[-3:]}" \
            if not self.overwrite else self.path

        # required states
        state_dict = {
            "model_params": self.model.state_dict(),
            "optimizer_params": self.optimizer.state_dict() \
                if self.optimizer is not None else None, 
            "scheduler_params": self.scheduler.state_dict() \
                if self.scheduler is not None else None,
            "frequency": self.frequency,
            "path": self.path,
            "overwrite": self.overwrite,
            "epoch": self.epoch
            }
        
        # additional states
        self.dict = dict(state_dict, **kwargs)
        torch.save(self.dict, path)
            
    def load_checkpoint(self, path, verbose=True, device=None): 

        """
        Loads the checkpoint from the specified path and sets the states of the model, optimizer, and scheduler.

        Args:
            path (str): The path where the checkpoint is stored.
            verbose (bool, optional): If True, print a message after loading the checkpoint. Default is True.
            device (torch.device, optional): The device to map the loaded checkpoint. Default is None.
        """
        
        # load checkpoint & set modules
        checkpoint = torch.load(path, map_location=device) 
        self.model.load_state_dict(checkpoint["model_params"])  
        self.frequency = checkpoint["frequency"]
        self.path = checkpoint["path"]
        self.epoch = checkpoint["epoch"]
        self.overwrite = checkpoint["overwrite"]
        self.dict = checkpoint

        # conditinal modules
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_params"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_params"])
            self.scheduler.optimizer = self.optimizer  

        if verbose:
            print("Checkpoint loaded")

    def __getitem__(self, item):

        """
        Allows retrieval of model, optimizer, scheduler, and additional states via indexing.

        Args:
            item (str): The key of the state to be retrieved.

        Returns:
            any: The state corresponding to the provided key.
        """

        if item == "model":
            return self.model
        if item == "optimizer":
            return self.optimizer
        if item == "scheduler":
            return self.scheduler
        return self.dict[item]