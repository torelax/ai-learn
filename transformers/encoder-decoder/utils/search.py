import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils.functional import generate_pad_mask

class DecoderSearch:

    """
    Base class for Decoder search strategies. Must be subclassed to create specific search strategies.
    """
        
    def search(self, *args, **kwargs):

        """
        Method stub for search method. Must be implemented in subclasses.
        """
        pass

    def __call__(self, *args, **kwargs):

        """
        Calls the search method when the object instance is called as a function.
        """

        return self.search(*args, **kwargs)

class Greedy(DecoderSearch):

    """
    Greedy search strategy for a Decoder.

    Attributes:
        sos (int): The id of the start of sentence token.
        eos (int): The id of the end of sentence token.
        maxlen (int): The maximum length of the generated sequences.
        alpha (float): The length penalty factor. Default is 0.6.
        eps (float): A small constant to ensure numerical stability. Default is 1e-9.
        base (torch.Tensor): A tensor used for computation.
    """


    def __init__(self, sos, eos, maxlen, alpha=0.6, eps=1e-9) -> None:
        super().__init__()
        self.sos = sos
        self.eos = eos
        self.maxlen = maxlen
        self.alpha = alpha
        self.eps = eps
        self.base = torch.tensor([[1]]).float()

    def search(self, ids, model, early_stop=False, device=None):

        """
        Performs greedy search over the sequences.

        Args:
            ids (torch.Tensor): The source sequences.
            model (model.transformer.Transformer): The transformer model used for the predictions.
            early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search. Default is False.
            device (torch.device, optional): The device to move tensors for computation. Default is None.

        Returns:
            Tuple[torch.Tensor, float]: The generated sequences and their corresponding batch score.
        """

        sos, eos, maxlen, alpha, eps, base = \
             self.sos, self.eos, self.maxlen, self.alpha, self.eps, self.base
        
        # handle sequence length
        if ids.size(1) > maxlen:
            raise ValueError(f"Can't process sequences with length greater than {maxlen}")
        
        maxlen = ids.size(1) # set maxlen for general translation
        
        # create/generate required tensors
        src = ids.long()
        trg = torch.tensor([[sos]]).long()
        trg = trg.repeat(src.size(0), 1)
        base = base.repeat(src.size(0), 1)
        mask = generate_pad_mask(src, model.pad_id)
        # move tensors to device
        src = src.to(device)
        trg = trg.to(device)
        mask = mask.to(device)
        base = base.to(device)
        # run greedy search
        beam = (trg, base)
        sequence, score = greedy_search(model, src, beam, eos, maxlen, alpha=alpha, 
                                        mask=mask, eps=eps, early_stop=early_stop)
        return sequence, score

def greedy_search(model, input, beam, eos, maxlen, mask=None, alpha=0.6, eps=1e-9, early_stop=False):

    """
    Function to perform greedy search.

    Args:
        model (model.transformer.Transformer): The transformer model used for the predictions.
        input (torch.Tensor): The source sequences.
        beam (Tuple[torch.Tensor, torch.Tensor]): The beam used for the search.
        eos (int): The id of the end of sentence token.
        maxlen (int): The maximum length of the generated sequences.
        mask (torch.Tensor, optional): The mask tensor for the input. Default is None.
        alpha (float, optional): The length penalty factor. Default is 0.6.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-9.
        early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search. Default is False.

    Returns:
        Tuple[torch.Tensor, float]: The generated sequences and their corresponding batch score.
    """

    model.eval()
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():

        # autoregress until sequence is complete
        output, score = beam
        while output.size(1) < maxlen:
            # predict & concatenate tokens
            out = model(input, output, src_mask=mask)
            pred = softmax(out)[:, -1]
            probs, tokens = torch.topk(pred, 1)
            output, score = torch.cat((output, tokens), dim=-1), torch.cat((score, probs), dim=-1)
            if skip(output, maxlen, eos, early_stop):
                break
        return output, log_score(score, alpha, eps)

class Beam(DecoderSearch):

    """
    Beam search strategy for a Decoder.

    Attributes:
        sos (int): The id of the start of sentence token.
        eos (int): The id of the end of sentence token.
        maxlen (int): The maximum length of the generated sequences.
        width (int): The beam width. Default is 3.
        alpha (float): The length penalty factor. Default is 0.6.
        eps (float): A small constant to ensure numerical stability. Default is 1e-9.
        fast (bool): If True, the beam tree pruning is performed faster. Default is False.
        base (torch.Tensor): A tensor used for computation.
    """

    def __init__(self, sos, eos, maxlen, width=3, alpha=0.6, eps=1e-9, fast=False):
        super().__init__()
        self.sos = sos
        self.eos = eos
        self.maxlen = maxlen
        self.width = width
        self.fast = fast
        self.alpha = alpha
        self.eps = eps
        self.base = torch.tensor([[1]]).float()

    def search(self, ids, model, early_stop=False, device=None):

        """
        Performs beam search over the sequences.

        Args:
            ids (torch.Tensor): The source sequences.
            model (model.transformer.Transformer): The transformer model used for the predictions.
            early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search. Default is False.
            device (torch.device, optional): The device to move tensors for computation. Default is None.

        Returns:
            Tuple[torch.Tensor, float]: The generated sequences and their corresponding batch score.
        """

        sos, eos, maxlen, width, fast, alpha, eps, base = \
            self.sos, self.eos, self.maxlen, self.width, self.fast, self.alpha, self.eps, self.base
        
        # handle sequence length
        if ids.size(1) > maxlen:
            raise ValueError(f"Can't process sequences with length greater than {maxlen}")
        
        maxlen = ids.size(1) + 2 # set maxlen for general translation
                
        # create/generate tensors
        src = ids.long()
        trg = torch.tensor([[sos]]).long()
        trg = trg.repeat(src.size(0), 1)
        base = base.repeat(src.size(0), 1)
        mask = generate_pad_mask(src, model.pad_id)
        # move tensors to device
        src = src.to(device)
        trg = trg.to(device)
        mask = mask.to(device)
        base = base.to(device)
        # run beam search
        beams = [(trg, base)]
        sequence, score = beam_search(model, src, beams, maxlen, eos, width=width, mask=mask, 
                                      alpha=alpha, eps=eps, fast=fast, early_stop=early_stop)
        return sequence, score

def beam_search(model, input, beams, maxlen, eos, width=3, mask=None, alpha=0.6, eps=1e-9, fast=False, early_stop=False):

    """
    Function to perform beam search.

    Args:
        model (model.transformer.Transformer): The transformer model used for the predictions.
        input (torch.Tensor): The source sequences.
        beams (List[Tuple[torch.Tensor, torch.Tensor]]): The list of beams used for the search.
        maxlen (int): The maximum length of the generated sequences.
        eos (int): The id of the end of sentence token.
        width (int, optional): The beam width. Default is 3.
        mask (torch.Tensor, optional): The mask tensor for the input. Default is None.
        alpha (float, optional): The length penalty factor. Default is 0.6.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-9.
        fast (bool, optional): If True, the beam tree pruning is performed faster. Default is False.
        early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search. Default is False.

    Returns:
        Tuple[torch.Tensor, float]: The generated sequences and their corresponding batch score.
    """

    model.eval()
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():

        # search up to maximum length
        for _ in range(maxlen):
            new_beams = []
            for beam in beams:
                output, score = beam
                if skip(output, maxlen, eos, early_stop):
                    new_beams.append(beam)
                    continue

                # predict & concatenate tokens
                out = model(input, output, src_mask=mask)
                pred = softmax(out)[:, -1]
                probs, tokens = torch.topk(pred, width)
                for i in range(width):
                    token, prob = tokens[:, i].unsqueeze(1), probs[:, i].unsqueeze(1)
                    new_output, new_score = torch.cat((output, token), dim=-1), torch.cat((score, prob), dim=-1)
                    new_beams.append((new_output, new_score))
                     
            # prune beam tree to contain best beams
            beams = sort_beams(new_beams, width, model.pad_id, alpha, eps, fast)
            if complete(beams, maxlen, eos, early_stop):
                break
        return beams[0][0], log_score(beams[0][1], alpha, eps)

def skip(sequences, maxlen, eos, early_stop):

    """
    Checks if all the sequences have reached the maximum length or contain the end of sentence token.

    Args:
        sequences (torch.Tensor): The sequences.
        maxlen (int): The maximum length of the sequences.
        eos (int): The id of the end of sentence token.
        early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search.

    Returns:
        bool: True if all sequences have reached the maximum length or contain the end of sentence token, False otherwise.
    """

    # determine if sequences are complete (i.e. maximum length reached or eos hit for all sequences)
    return (torch.all(torch.any(torch.eq(sequences, eos), dim=-1)).item() and early_stop) or \
            sequences.size(1) == maxlen
        
def complete(beams, maxlen, eos, early_stop):

    """
    Checks if all the beams are complete.

    Args:
        beams (List[torch.Tensor, torch.Tensor]): The list of beams.
        maxlen (int): The maximum length of the sequences.
        eos (int): The id of the end of sentence token.
        early_stop (bool): If True, a sequence that hits the eos token will be marked as complete during search.

    Returns:
        bool: True if all beams are complete, False otherwise.
    """

    # determine if all beams are complete
    return all([skip(seq, maxlen, eos, early_stop) for seq, scores in beams])

def sort_beams(beams, width, pad_id, alpha=0.6, eps=1e-9, fast=False):

    """
    Sorts and prunes the beam tree.

    Args:
        beams (List[torch.Tensor, torch.Tensor]): The list of beams.
        width (int): The beam width.
        pad_id (int): The id of the padding token.
        alpha (float, optional): The length penalty factor. Default is 0.6.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-9.
        fast (bool, optional): If True, the beam tree pruning is performed faster. Default is False.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: The sorted and pruned beams.
    """

    # sort based on batch score (faster)
    if fast:
        return sorted(beams, reverse=True, key=lambda x: log_score(x[1], alpha, eps))[:width]
    
    # sort based on best score for each sequence in a batch (slower)
    new_beams = []
    batches = [([], []) for _ in range(width)]

    # go through batches in beams
    for i in range(beams[0][0].size(0)):
        scores = []
        for output, score in beams:
            seq, seq_socre = output[i], score[i]
            scores.append((seq, seq_socre))

        # keep the best sequences from each row in beams
        scores = sorted(scores, reverse=True, key=lambda x: log_score(x[1].unsqueeze(0), alpha, eps))[:width]
        for j, beam in enumerate(scores):
            seq, seq_score = beam
            batches[j][0].append(seq)
            batches[j][1].append(seq_score)
            
    # convert list of tensors to batch of tensors
    for seqs, seq_scores in batches:
        output = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        score = pad_sequence(seq_scores, batch_first=True, padding_value=1.0 - eps)
        new_beams.append((output, score))
    return new_beams
                    
def log_score(tensor, alpha=0.6, eps=1e-9):

    """
    Computes the average log score for batched tensors.

    Args:
        tensor (torch.Tensor): The tensor.
        alpha (float, optional): The length penalty factor. Default is 0.6.
        eps (float, optional): A small constant to ensure numerical stability. Default is 1e-9.

    Returns:
        float: The average log score.
    """

    # find the average log score for batched tensors
    norm = np.power(tensor.size(1), alpha)
    score = torch.sum(torch.log(tensor + eps), dim=-1, keepdim=True) / norm
    likelihoods = torch.exp(score)
    n = tensor.size(0)
    total = torch.sum(likelihoods, dim=0).item()
    avg = total / n
    return avg