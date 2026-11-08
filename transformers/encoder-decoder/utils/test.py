import torch
import torch.nn as nn
from utils.functional import generate_masks

def test(dataloader, model, device=None):    

    """
    Evaluates a model over a dataloader.

    Args:
        dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
        model (model.transformer.Transformer): The transformer model to be evaluated.
        device (torch.device, optional): The device to move tensors for computation. Defaults to None.

    Returns:
        float: The average cross entropy loss of the model on the dataloader.
    """

    model.eval()
    m = len(dataloader)
    cross_entropy = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    net_loss = 0
    # test model over dataloader
    with torch.no_grad():
        
        for batch in dataloader:            
            inputs, labels = batch.src, batch.trg
            src, trg, out = inputs, labels[:, :-1], labels[:, 1:] 
            src, trg, out = src.long(), trg.long(), out.long()
            src_mask, trg_mask = generate_masks(src, trg, model.pad_id)
            # move tensors to device
            src, trg, out = src.to(device), trg.to(device), out.to(device)
            src_mask, trg_mask = src_mask.to(device), trg_mask.to(device)
            # generate prediction
            pred = model(src, trg, src_mask=src_mask, trg_mask=trg_mask) 
            pred, out = pred.contiguous().view(-1, pred.size(-1)), out.contiguous().view(-1) 
            # calculate loss
            loss = cross_entropy(pred, out)
            net_loss += loss.item()
        # calculate average loss
        loss = net_loss / m
        return loss
    
def predict(dataloader, model, search, device=None):

    """
    Generates predictions for a dataloader using a model and search strategy.

    Args:
        dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
        model (model.transformer.Transformer): The transformer model to be used for predictions.
        search (utils.search.DecoderSearch): The search strategy to be used for prediction.
        device (torch.device, optional): The device to move tensors for computation. Defaults to None.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples where the first element is the output 
                                                sequences and the second element is the labels.
    """

    predictions = []
    for batch in dataloader:
        inputs, labels = batch.src, batch.trg
        # generate output from search
        outputs, score = search.search(inputs, model, early_stop=True, device=device)       
        predictions.append((outputs, labels))
    return predictions

def prompt(sequence, model, tokenizer, search, early_stop=False, device=None):    

    """
    Prompts the user to enter a sequence of text and returns the model's translation of it.

    Args:
        model (model.transformer.Transformer): The transformer model to be used for predictions.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the data.
        search (utils.search.DecoderSearch): The search strategy to be used for prediction.
        device (torch.device, optional): The device to move tensors for computation. Defaults to None.

    Returns:
        str: The model's translation of the input text.
    """  

    # get input sequence & convert to tensor
    sequence = sequence.strip()
    ids = tokenizer.encode(sequence, special_tokens=False, module="source")
    ids = tokenizer.truncate(ids, model.maxlen, end=True)
    ids = torch.tensor(ids).contiguous()
    # generate output from input sequence
    sequence, score = search.search(ids, model, early_stop=early_stop, device=device)
    # convert back to string
    translation = tokenizer.decode(sequence.tolist(), special_tokens=False, module="target")[0]
    return translation