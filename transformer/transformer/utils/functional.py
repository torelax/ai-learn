import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.quantize import quantize_model, get_quantized_dtype
from torch.ao.nn.quantized.modules.linear import LinearPackedParams    
from torch.ao.nn.quantized.modules.embedding_ops import EmbeddingPackedParams    

def torchset_to_samples(torch_dataset):

    """
    Converts a PyTorch Dataset to lists of inputs & labels.

    Args:
        torch_dataset (torchtext.datasets.TranslationDataset): The PyTorch Translation Dataset to convert.

    Returns:
        Tuple[List[str], List[str]]: Two lists representing inputs and labels.
    """

    # convert PyTorch Dataset to lists of inputs & labels
    inputs, labels = [], []
    for example in torch_dataset:
        src, trg = example.src, example.trg
        inputs.append(src)
        labels.append(trg)
    return inputs, labels

def parse_config(path, encoding="utf-8", verbose=True):

    """
    Parses a configuration file into a dictionary.

    Args:
        path (str): The path to the configuration file.
        encoding (str, optional): The encoding of the configuration file. Default is 'utf-8'.

    Returns:
        dict: The dictionary containing the configuration parameters and their values.
    """

    config = dict()
    # open config file
    with open(path, "r", encoding=encoding) as file:
        contents = file.readlines()
        for line in contents:
            line = line.strip().split("=")
            if len(line) != 2:
                continue
            # get parameter & argument
            param, value = line
            param, arg = param.strip(), str_to_arg(value.strip())
            # store parameter & argument
            config[param] = arg

    if verbose:
        print("Config parsed")
    return config

def save_config(config_path, path, verbose=True):

    """
    Writes the contents of a configuration file to another file.

    Args:
        config_path (str): The path to the original configuration file.
        path (str): The path to the new configuration file.
        verbose (bool, optional): If True, prints a message after saving. Default is True.
    """

    # write config.py file to path
    path = "config.txt" if path is None else path
    with open(config_path) as file:
        lines = [line.rstrip() for line in file]
        for overwrite, line in enumerate(lines):
            write(line, path=path, overwrite=not overwrite)
            
    if verbose:
        print(f"Config saved")

def str_to_arg(value):

    """
    Converts a string to an appropriate Python type.

    Parameters:
        value (str): to be converted.

    Returns:
        any: corresponding Python value.
    """

    value = value.lower().capitalize()
    if value.isnumeric():
        return int(value)
    if value.replace("-", "").replace("e", "").isnumeric() or \
        value.replace(".", "").isnumeric():
        return float(value)
    if value[0] == "(" and value[-1] == ")":
        val1, val2 = value[1:-1].split(",")
        return float(val1), float(val2)
    if value == "True":
        return True
    if value == "False":
        return False
    return None

def read_data(path, size=None, shuffle=False, encoding="utf-8"):

    """
    Reads a file into a list up to a certain size.

    Parameters:
        path (string): The path to the file.
        size (int, optional): The max number of lines to pull. Default None.
        shuffle (bool, optional): Shuffles the data if True. Default is False.
        encoding (str, optional): The encoding of the file Default is 'utf-8'.

    Returns:
        List[str]: The list of lines from the file.
    """

    # read a file (.txt filetype) into a list

    with open(path, mode="r", encoding=encoding) as file:
        data = file.readlines()
        data = [line.strip() for line in data] # remove newlines
        size = len(data) if size is None else size

        # shuffle (if applicable)
        if shuffle:
            np.random.shuffle(data)
        return data[:size] # return specified slice
    
def write_data(data, path, size=None, encoding="utf-8"):

    """
    Writes a list of data to a file up to a certain size.

    Parameters:
        data (list): The list of strings to be written to the file.
        path (str): The path to the file
        size (int, optional): The max number of lines to write. Default is None.
        encoding (str, optional): The encoding of the file. Default is 'utf-8'.
    """

    # write a list of data to a file (.txt filetype) up to a certain size
    size = len(data) if size is None else size
    for i in range(size):
        write(data[i].strip(), path, overwrite=False, 
              encoding=encoding)
        
def printer(loss=None, norm=None, lr=None, lr_round=3, epoch=None, warmup=None, clock_info=None, 
            test_info=None, saved=None, print_output=False):

    """
    Prints various information during the model's training/testing process.

    Args:
        loss (float, optional): The loss to pass from the model. Default is None.
        norm (float, optional): The current gradient norm. Default is None.
        lr (float, optional): The current learning rate of the optimizer. Default is None.
        lr_round (int, optional): Number of decimal places to round the learning rate to. Default is 3.
        epoch (int, optional): The current training epoch. Default is None.
        warmup (bool, optional): True if the epoch was a warmup step, False otherwise. Default is None.
        clock_info (Tuple[str, str], optional): Tuple containing two strings. The first string represents the epoch duration, and the second string represents the total elapsed time. Default is None.
        test_info (Tuple[float, float], optional): Tuple containing two floats. The first float is the test loss, and the second float is the BLEU score. Default is None.
        saved (bool, optional): True if a checkpoint was saved, False otherwise. Default is None.
        print_output (bool, optional): If True, the output will be printed. Default is False.

    Returns:
        str: Formatted string of the metrics.
    """

    # header
    heading = f"Epoch {epoch}" if epoch else \
                "Overview"
    output = create_block(heading, length=67, upper=True, lower=True)

    # time
    if clock_info is not None:
        time_info = f"Epoch Duration: {clock_info[0]} | Elapsed Time: {clock_info[1]}" if epoch else \
                f"Elapsed Time: {clock_info[1]}"
        output += create_block(time_info, length=67)

    # metrics
    loss_info = f"Train Loss: {loss:.4f}" if epoch else \
                f"Avg Train Loss: {loss:.4f}"
    if test_info is not None:
        loss_info += f" | Test Loss: {test_info[0]:.4f}" if epoch else \
                        f" | Avg Test Loss: {test_info[0]:.4f}"
        loss_info += f" | BLEU: {test_info[1]:.1f}" if epoch else \
                    f" | Best BLEU: {test_info[1]:.1f}"
    output += create_block(loss_info, length=67)

    # optimization
    optim_info = ""
    if norm is not None and lr is not None:
        optim_info = f"Gradient Norm: {norm:.1f} | Learning Rate: {format_lr(lr, lr_round)}"
    elif norm is not None:
        optim_info += f"Gradient Norm: {norm:.1f}"
    elif lr is not None:
        optim_info += f"Learning Rate: {format_lr(lr, lr_round)}"
    output += create_block(optim_info, length=67)

    # other info
    other_info = ""
    if warmup is not None and saved is not None:
        other_info = f"Warmup Step: {warmup} | Checkpoint Saved: {saved}"
    elif warmup is not None:
        other_info = f"Warmup Step: {warmup}"
    elif saved is not None:
        other_info = f"Checkpoint Saved: {saved}"
    output += create_block(other_info, length=67, lower=True)[:-1] # cuts "\n"

    # add bottom border
    if print_output:
        print(output)
    return output

def format_lr(lr, places=3):

    """
    Formats a learning rate float for printing.

    Args:
        lr (float): The learning rate.
        places (int): The number of decimal places to round to. Default is 3.

    Returns:
        str: The formatted learning rate string.
    """

    lr_format = f"{lr}"
    # find where float values end
    stop = lr_format.find("e")

    # too many floating values
    if len(lr_format[:stop]) - 2 > places:
        # pull the required digits (including rounding)
        val = float(lr_format[:places + 3])
        val = str(round(val, places)) # round
        # recombine w/ scientific notation suffix
        lr_format = val + lr_format[stop:]
    return lr_format

def create_block(info, length, upper=False, lower=False):

    """
    Creates a string block of a specific length with possible upper and lower borders.

    Args:
        info (str): The information to be included in the block.
        length (int): The length of the block. 
        upper: (bool): Includes upper border if True. Default is False.
        lower: (bool): Includes lower border if True. Default is False.

    Returns:
        str: The string of the formatted block.
    """

    div = f"{'-' * length}"
    block = f"|{f'{info}'.center(length - 2)}|\n" if info else ""
    # add borders (if applicable)
    if upper:
        block = f"{div}\n{block}"
    if lower:
        block += f"{div}\n"
    return block

def generate_pad_mask(seq, pad_id):

    """
    Creates a padding mask for the sequence.

    Args:
        seq (torch.Tensor): Input sequence.
        pad_id (int): The id used for the pad token.

    Returns:
        torch.Tensor: Padding mask for the input sequence.
    """

    # create pad mask to ignore calculating gradients where there's pad tokens
    mask = (seq != pad_id).unsqueeze(-2)
    return mask

def generate_nopeak_pad_mask(trg, pad_id):

    """
    Creates a mask to prevent the Decoder from looking ahead.

    Args:
        trg (torch.Tensor): Target sequence.
        pad_id (int): The id used for the pad token.

    Returns:
        torch.Tensor: Mask for the target sequence.
    """

    # create a subsequent mask to prevent Decoder peaking ahead to predict tokens
    trg_mask = generate_pad_mask(trg, pad_id)
    trg_len = trg.size(1)
    trg_nopeak_mask = torch.triu(torch.ones((1, trg_len, trg_len)) == 1)
    trg_nopeak_mask = trg_nopeak_mask.transpose(1, 2)
    trg_mask = trg_mask & trg_nopeak_mask
    return trg_mask

def generate_masks(src, trg, pad_id):

    """
    Creates required masks for the source and target sequences.

    Args:
        src (torch.Tensor): Source sequence.
        trg (torch.Tensor): Target sequence.
        pad_id (int): The id used for the pad token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Masks for the source and target sequences.
    """

    # create required masks for source & target sequences for training
    src_mask = generate_pad_mask(src, pad_id)
    trg_mask = generate_nopeak_pad_mask(trg, pad_id)
    return src_mask, trg_mask

def parameter_count(model):

    """
    Returns the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: Number of trainable parameters in millions.
    """

    # total registered parameters
    total_params = sum([param.numel() for param in model.parameters() if param.requires_grad])


    # account for quantized parameters
    for module in model.modules():

        if isinstance(module, EmbeddingPackedParams):
            weights = module._weight()
            total_params += weights.numel()

        elif isinstance(module, LinearPackedParams):
            weights, bias = module._weight_bias()
            total_params += weights.numel()
            if bias is not None:
                total_params += bias.numel()

    return total_params / 1e6 # parameters (in millions)

def model_size(model):

    """
    Returns the size of the model in MB.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: Size of the model in MB.
    """

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_bytes = buffer.tell()
    model_size_mb = model_size_bytes / 1e6
    return model_size_mb

def grad_norm(model, p=2):

    """
    Computes the gradient norm of the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        p (int, optional): Norm type. Defaut is 2.

    Returns:
        torch.Tensor: Gradient norm of the model.
    """

    grads = [param.grad.detach().flatten() for param in model.parameters() \
             if param is not None]
    norm = torch.cat(grads).norm(p)
    return norm

def graph(losses, test_losses, bleus, path=None):

    """
    Generates a plot of training metrics.

    Args:
        losses (List[float]): List of training losses.
        bleus (List[float]): List of BLEU scores.
        test_losses (List[float]): List of validation losses.
        path (str, optional): Filepath to save the plot. Defaut is None.
    """

    # graph metrics of model
    epochs = list(range(len(losses)))
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=1)

    # loss subplot
    axs[0].plot(epochs, losses, color="red", label="train loss")
    axs[0].plot(epochs, test_losses, color="blue", label="test loss")
    axs[0].set_title(f"Training {'& Validation' if test_losses is not None else ''} Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")

    # bleu subplot
    axs[1].plot(epochs, bleus, color="orange", label="BLEU")
    axs[1].set_title("BLEU Scores")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("BLEU")

    # create path (if non-existent)
    if path is not None:
        create_path(path)
        fig.savefig(fname=path)
    plt.show()

def save_model(model, path=None, verbose=True):

    """
    Saves the model state to a .pt file.

    Args:
        model (torch.nn.Module): The PyTorch model.
        path (str, optional): The filepath to save the model. Defaut is 'model.pt'.
        verbose (bool, optional): If True, prints a message after saving. Defaut is True.
    """

    # save model to path
    path = "model.pt" if path is None else path
    create_path(path)
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"Model params saved")

def load_model(model, path, verbose=True, device=None):

    """
    Loads the model state from a .pt file.

    Args:
        model (torch.nn.Module): The PyTorch model.
        path (str): The filepath to load the model from.
        verbose (bool, optional): If True, prints a message after loading. Defaut is True.
        device (torch.device, optional): The device to load the model to. Defaut is None.

    Returns:
        torch.nn.Module: The model with the loaded state.
    """

    
    # handle quantization
    params = torch.load(path, map_location=device)
    quantize_dtype = get_quantized_dtype(params)
    if quantize_dtype is not None:
        quantize_model(model, dtype=quantize_dtype, inplace=True)
    # load model from path to device
    model.load_state_dict(params)
    if verbose:
        print(f"Model params loaded")
    return model

def save_module(module, path=None, verbose=True):

    """
    Saves the module to a .pt file.

    Args:
        module (Any): The PyTorch module.
        path (str, optional): The filepath to save the module. Defaut is 'module.pt'.
        verbose (bool, optional): If True, prints a message after saving. Defaut is True.
    """

    # save module to path
    path = "module.pt" if path is None else path
    create_path(path)
    torch.save(module, path)
    if verbose:
        print("Module saved")

def load_module(path, verbose=True):

    """
    Loads the module from a .pt file.

    Args:
        path (str): The filepath to load the module from.
        verbose (bool, optional): If True, prints a message after loading. Defaut is True.

    Returns:
        Any: The loaded module.
    """

    # load module from path
    module = torch.load(path)
    if verbose:
        print("Module loaded")
    return module

def create_path(path):

    """
    Creates a directory if it doesn't already exist.

    Args:
        path (str): The directory path to be created.
    """

    # create a path (for non-existent paths)
    path = path.split("/")
    path = "/".join(path[:-1]) + "/"
    if path and not os.path.exists(path):
        os.makedirs(path)

def write(contents, path, overwrite=False, encoding="utf-8"):

    """
    Writes contents to a file.

    Args:
        contents (str): The content to be written to the file.
        path (str): The filepath to write the contents to.
        overwrite (bool, optional): If True, overwrites the existing file. If False, appends to the existing file. Defaut is False.
        encoding (str, optional): The encoding of the file. Defaut is 'utf-8'.
    """

    # write contents (strings) to path
    create_path(path)
    contents += "\n"
    arg = "w" if overwrite else "a"
    file = open(path, arg, encoding=encoding)
    file.write(contents)
    file.close()      