import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig, default_dynamic_qconfig, \
                                            float16_dynamic_qconfig
                                       
def quantize_model(model, dtype=torch.qint8, inplace=False):

    """
    Quantizes the given model based on the specified data type (dtype).

    Args:
        model (model.transformer.Transformer): The transformer model to be quantized.
        dtype (torch.dtype, optional): The data type to quantize the model to. Options include torch.qint8, torch.float16, 
                                        and torch.half. Default is torch.qint8.
        inplace (bool, optional): If True, quantize the model in place, otherwise return a new quantized model. Default is False.

    Returns:
        model.transformer.Transformer: The quantized transformer.

    Raises:
        ValueError: If an unsupported dtype is provided.
    """

    if dtype == torch.qint8:
        q_config = {
                    nn.Embedding: float_qparams_weight_only_qconfig,
                    nn.Linear: default_dynamic_qconfig
                    }
        
    elif dtype == torch.float16 or torch.half:
        q_config = {
                    nn.Embedding: float_qparams_weight_only_qconfig,
                    nn.Linear: float16_dynamic_qconfig
                    }
    else:
        raise ValueError(f"This dtype is unsopported {dtype}")
    
    quantized_model = quantize_dynamic(model.to("cpu"), qconfig_spec=q_config, dtype=dtype, inplace=inplace)
    return quantized_model

def get_quantized_dtype(state_dict):

    """
    Retrieves the quantized data type (dtype) from a given state dictionary. If the dtype is found to be either torch.qint8 or torch.float16, 
    the tensors in the state dictionary are moved to the CPU using the move_quantized_tensors function.

    Args:
        state_dict (dict): A state dictionary containing values that may include torch.dtype.

    Returns:
        torch.dtype or None: The quantized dtype if found (either torch.qint8 or torch.float16), otherwise None.

    Note:
        If the quantized dtype is found, this function modifies the input state_dict in place by moving the tensors to the CPU.
    """

    dtype = None
    for value in state_dict.values():
        if isinstance(value, torch.dtype):
            if value == torch.qint8:
                dtype = torch.qint8
                break
            elif value == torch.float16:
                dtype = torch.float16
                break

    if dtype is not None:
        move_quantized_tensors(state_dict)
        return dtype
    return None

def move_quantized_tensors(state_dict):

    """
    Moves the tensors in a given state dictionary to the CPU. If the value is a tuple of tensors, 
    it handles them using the handle_tupled_tensors function.

    Args:
        state_dict (dict): A state dictionary containing keys and values, 
        where values may be tensors or tuples of tensors.

    Note:
        This function modifies the input state_dict in place.
    """

    for key, value in state_dict.items():

        if isinstance(value, torch.Tensor):
            state_dict[key] = value.to("cpu")
        elif isinstance(value, tuple):
            tupled_tesnor = handle_tupled_tensors(value)
            state_dict[key] = tupled_tesnor

def handle_tupled_tensors(tupled_tensor):

    """
    Handles a tuple of tensors, moving them to the CPU if they are not None.

    Args:
        tupled_tensor (Tuple[torch.Tensor]): A tuple of tensors that may include None values.

    Returns:
        Tuple[torch.Tensor]: A tuple of tensors with the same content as the input, but moved to the CPU if not None.
    """

    pair = []
    for tensor in tupled_tensor:

        if tensor is not None:
            tensor = tensor.to("cpu")
        pair.append(tensor)

    return tuple(pair)