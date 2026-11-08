import torch 
import numpy as np

class DataLoader:

    """
    A class used to batchify and handle source and target sequences.

    Args:
        source (List[str]): The list of source sequences.
        target (List[str]): The list of target sequences.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.
        maxlen (int, optional): The maximum length of a sequence. Default is None.
        batch_size (int, optional): The size of each batch. Default is 32.
        shuffle (bool, optional): If True, shuffle the batches. Default is False.
        drop_last (bool, optional): If True, drop the last batch if it is not of the same size as the others. Default is False.

    Attributes:
        source (List[str]): The list of source sequences.
        target (List[str]): The list of target sequences.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.
        maxlen (int): The maximum length of a sequence.
        batch_size (int): The size of each batch.
        shuffle (bool): If True, shuffle the batches.
        drop_last (bool): If True, drop the last batch only if it's not of the same size as the others.
        batches (List[Batch]): The list of batches.
    """

    def __init__(self, source, target, tokenizer, maxlen=None, batch_size=32, shuffle=False, drop_last=False):

        assert(len(source) == len(target))
        self.source = source
        self.target = target
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batches = self.batchify(source, target, tokenizer, maxlen=maxlen, batch_size=batch_size, 
                                     shuffle=shuffle, drop_last=drop_last)

    def batchify(self, source, target, tokenizer, maxlen, batch_size, shuffle, drop_last):

        """
        Creates batches of sequences based on similar lengths.

        Args:
            source (List[str]): The list of source sequences.
            target (List[str]): The list of target sequences.
            tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.
            maxlen (int): The maximum length of a sequence.
            batch_size (int): The size of each batch.
            shuffle (bool): If True, shuffle the batches.
            drop_last (bool): If True, drop the last batch only if it's not of the same size as the others.

        Returns:
            List[Batch]: The list of batches.
        """

        # batch sequences based on similar lengths
        bacthes = []
        pairs = self.parify(source, target, maxlen)
        for i in range(0, len(pairs), batch_size):
            samples = pairs[i: i + batch_size]
            bacthes.append(self.batch(samples, tokenizer, shuffle))
            
        # drop last batch and/or shuffle (if applicable) 
        if drop_last and len(source) % batch_size != 0:
            bacthes.pop()
        if shuffle:
            np.random.shuffle(bacthes)
        return bacthes

    def batch(self, samples, tokenizer, shuffle):

        """
        Creates a batch of sequences with the same length.

        Args:
            samples (List[Tuple[str, str]]): The list of samples to batchify.
            tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.
            shuffle (bool): If True, shuffle the samples.

        Returns:
            Batch: A batch of the samples.
        """

        # create a batch of sequences with the same length
        maxlen = length(max(samples, key=length))
        batch = [tuple(tokenizer.pad(sample, maxlen, end=True)) for sample in samples]
        if shuffle:
            np.random.shuffle(batch)
        return Batch(batch)

    def parify(self, source, target, maxlen):

        """
        Pairs source and target sequences (ignores those exceeding maxlen).

        Args:
            source (List[str]): The list of source sequences.
            target (List[str]): The list of target sequences.
            maxlen (int): The maximum length of a sequence.

        Returns:
            List[Tuple[str, str]]: The list of pairs sorted based on length (ascending).
        """

        # pair source & target sequences (ignore those exceeding maxlen)
        maxlen = float("inf") if maxlen is None else maxlen
        pairs = [(src, trg) for src, trg in zip(source, target) if \
                 len(src) <= maxlen and len(trg) <= maxlen]
        # sort sequences based on length
        return sorted(pairs, key=length)
    
    def size(self):

        """
        Gets the total number of sentence pairs.

        Returns:
            int: The total number of sentence pairs.
        """

        # get total number of sentence pairs
        return sum([batch.size(0) for batch in self.batches])

    def __iter__(self):

        """
        Returns an iterator for the batches.

        Returns:
            Iterator: The iterator for the batches.
        """
                
        return iter(batch for batch in self.batches)
    
    def __len__(self):

        """
        Gets the number of batches.

        Returns:
            int: The number of batches.
        """

        return len(self.batches)

    def __getitem__(self, item):

        """
        Retrieves a batch at the specified index.

        Args:
            item (int): The index of the batch.

        Returns:
            Batch: The batch at the specified index.
        """

        return self.batches[item]
    
    def __str__(self):

        """
        Returns a string representation of the DataLoader.

        Returns:
            str: A string representation of the DataLoader.
        """

        return str(self.batches)

class Batch:

    """
    A class used to represent a batch of sequences.

    Args:
        batch (List[Tuple[str, str]]): The batch of sentence pairs to be tensorized.

    Attributes:
        src (torch.Tensor): The tensor of source sequences.
        trg (torch.Tensor): The tensor of target sequences.
    """

    def __init__(self, batch):
        self.src = None
        self.trg = None
        self.tensify(batch)
        
    def tensify(self, batch):

        """
        Turns sequences into a PyTorch Tensor.

        Args:
            batch (List[Tuple[str, str]]): The batch of sentence pairs to be tensorized.
        """

        # turn sequences into a PyTorch tensor
        source, target = [], []
        for src, trg in batch:
            source.append(src)
            target.append(trg)
        self.src, self.trg = torch.tensor(source).long(), \
                            torch.tensor(target).long()
        
    def size(self, *args):

        """
        Gets the size of the source tensor. Reflexive of target tensor size.

        Args:
            *args: Variable length argument list.

        Returns:
            torch.Size: The size of the source tensor.
        """

        return self.src.size(*args)

    def __len__(self):

        """
        Gets the length of the source tensor. Reflexive of target tensor length.

        Returns:
            int: The length of the source tensor.
        """

        return self.src.size(0)
    
    def __str__(self):

        """
        Returns a string representation of the Batch.

        Returns:
            str: A string representation of the Batch.
        """

        return str((self.src, self.trg))
    
    def __getitem__(self, item):

        """
        Retrieves the source and target tensors at the specified index.

        Args:
            item (int): The index of the tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The source and target tensors at the specified index.
        """

        return self.src[item], self.trg[item]
    
def length(pair):

    """
    Gets the length of the longest sequence in a sentence pair.

    Args:
        pair (Tuple[str, str]): The pair of sentences.

    Returns:
        int: The length of the longest sentence in the pair.
    """

    # get length of largest sequence in sentence pair
    return len(max(pair, key=len))