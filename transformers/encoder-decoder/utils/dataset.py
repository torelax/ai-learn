import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from utils.dataloader import DataLoader

class Dataset(IterableDataset):

    """
    A class used to handle sequences of inputs and labels for machine learning tasks.

    Args:
        inputs (List[str]): The list of input sequences.
        labels (List[str]): The list of label sequences.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.

    Attributes:
        inputs (List[str]): The list of input sequences.
        labels (List[str]): The list of label sequences.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the sequences.
        size (int): The number of sequences.
        pointer (int): The current position in the sequence.
        indices (np.array): An array of shuffled indices for sampling the sequences.
    """

    def __init__(self, inputs, labels, tokenizer):
        if len(inputs) != len(labels):
            raise ValueError(f"The inputs and labels must have the same size (inputs size: {len(inputs)} != labels size: {len(labels)}).")
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.size = len(inputs)
        self.pointer = 0
        self.indices = np.random.choice(len(self), len(self), replace=False)

    def sample(self, n=1):

        """
        Samples n pairs of input and label sequences.

        Args:
            n (int, optional): The number of samples. Default is 1.

        Returns:
            List[Tuple[str, str]]: The list of sampled pairs of input and label sequences.
        """

        # sample n amount of input & label pairs
        if n > self.size or n < 0:
            raise ValueError(f"The value of 'n' must be in the range [0, {self.size}), not {n}.")

        inputs, labels = self.inputs, self.labels
        pointer, indices = self.pointer, list(self.indices)
        samples = [(inputs[i], labels[i]) for i in indices[pointer: pointer + n]]
        self.pointer += n 

        # resample (if necessary)
        if self.pointer > self.size - 1:
            self.pointer = 0
            np.random.shuffle(self.indices)
        return samples

    def dataframe(self, headers=None):

        """
        Creates a dataframe for the inputs and labels.

        Args:
            headers (Tuple[str], optional): The headers (i.e. column names) for the dataframe. Default is None.

        Returns:
            pd.DataFrame: The dataframe of the inputs and labels.
        """

        # create dataframe for inputs & labels
        if headers is None:
            headers = "inputs", "labels"
        d = {f"{headers[0]}": self.inputs, f"{headers[1]}": self.labels}
        return pd.DataFrame(d)

    def avgseq(self):

        """
        Gets the average length of the sequences.

        Returns:
            int: The average length of the sequences.
        """

        # get average length of sequences
        tokenizer, inputs, labels, size = self.tokenizer, self.inputs, self.labels, self.size
        tokenized_inputs, tokenized_labels = tokenizer.tokenize(inputs, special_tokens=False, module="source"), \
                                            tokenizer.tokenize(labels, special_tokens=True, module="target")
        m = sum(len(tokens) for tokens in tokenized_inputs)
        n = sum(len(tokens) for tokens in tokenized_labels)
        avg = (m + n) // (size * 2)
        return avg
    
    def maxseq(self):

        """
        Gets the maximum length of the sequences.

        Returns:
            int: The maximum length of the sequences.
        """

        # get largest length of sequences
        tokenizer, inputs, labels = self.tokenizer, self.inputs, self.labels
        tokenized_inputs, tokenized_labels = tokenizer.tokenize(inputs, special_tokens=False, module="source"), \
                                            tokenizer.tokenize(labels, special_tokens=True, module="target")
        corpus = tokenized_inputs + tokenized_labels
        maxlen = len(max(corpus, key=len))
        return maxlen

    def dataloader(self, maxlen=None, batch_size=32, shuffle=False, drop_last=False):

        """
        Creates a DataLoader for the tokenized inputs and labels.

        Args:
            maxlen (int, optional): The maximum length of a sequence allowed. Default is None.
            batch_size (int, optional): The size of each batch. Default is 32.
            shuffle (bool, optional): If True, shuffle the batches. Default is False.
            drop_last (bool, optional): If True, drop the last batch only if it's not of the same size as the others. Default is False.

        Returns:
            utils.dataloader.DataLoader: The DataLoader of the tokenized inputs and labels.
        """

        # create dataloader for tokenized (to ids) inputs & labels
        inputs, labels, tokenizer = self.inputs, self.labels, self.tokenizer
        source, target = tokenizer.encode(inputs, special_tokens=False, module="source"), \
                        tokenizer.encode(labels, special_tokens=True, module="target")
        dataloader = DataLoader(source, target, tokenizer, maxlen, batch_size=batch_size, 
                                shuffle=shuffle, drop_last=drop_last)
        return dataloader
    
    def __getitem__(self, index):

        """
        Retrieves the input and label at the specified index.

        Args:
            index (int): The index of the sequences.

        Returns:
            Tuple[str, str]: The input and label at the specified index.
        """

        return self.inputs[index], self.labels[index]

    def __iter__(self):

        """
        Returns an iterator for the inputs and labels.

        Returns:
            Iterator: The iterator for the inputs and labels.
        """

        return iter([(input, label) for input, label in zip(self.inputs, self.labels)])

    def __len__(self):

        """
        Gets the number of sequences.

        Returns:
            int: The number of sentence pairs.
        """

        return self.size

    def __str__(self):

        """
        Returns a string representation of the Dataset.

        Returns:
            str: A string representation of the Dataset.
        """

        return str(self.dataframe())