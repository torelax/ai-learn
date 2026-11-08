from tokenizers import decoders, models, normalizers, pre_tokenizers, \
                        processors, trainers, Tokenizer as _Tokenizer

default = {"sos": "<sos>", "eos": "<eos>", "unk": "<unk>", "pad": "<pad>", "mask": "<mask>"}

class Tokenizer:

    """
    Tokenizer for tokenizing, encoding, and decoding text.

    Args:
        prefix (str, optional): The prefix to be used in the tokenization process. Default is '##'
        special_tokens (dict, optional): A dictionary mapping special token names to their string values. 
                               Default is the `default` dictionary.

    Attributes:
        tokenizer (tokenizers.Tokenizer): The instance of the huggingface Tokenizer.
        prefix (str): The prefix to be used in the tokenization process. 
        special_tokens (dict): A dictionary mapping special token names to their string values.
        trained (bool): Whether the tokenizer has been trained.
    """    

    def __init__(self, prefix="##", special_tokens=default):
        self.tokenizer = self.init(prefix, special_tokens)
        self.prefix = prefix
        self.special_tokens = special_tokens
        self.trained = False

    def init(self, prefix, special_tokens):

        """
        Initializes all the required tokenizer attributes huggingface Tokenizer.

        Args:
            prefix (str): The prefix for sub-tokens.
            special_tokens (dict): The dictionary mapping of special tokens keywords a desired alias.

        Returns:
            tokenizers.Tokenizer: The defined huggingface Tokenizer.
        """

        # init basic tokenizer
        tokenizer = _Tokenizer(models.WordPiece(unk_token=special_tokens["unk"]))
        tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.Strip()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece(prefix=prefix)
        return tokenizer

    def train(self, corpus, size=25000, min_freq=0):

        """
        Trains the huggingface Tokenizer over a corpus.

        Args:
            corpus (List[str] or Iterator[str]): A list or iterator containing the entire corpus.
            size (int, optional): The maximum number of tokens the tokenizer can have. Default is 25000.
            min_freq (int, optional): the minimum frequency a token must have to be in the vocabulary. Default is 0.
        """

        # train tokenizer over corpus
        tokenizer, prefix, special_tokens = self.tokenizer, self.prefix, self.special_tokens
        trainer = trainers.WordPieceTrainer(vocab_size=size, show_progress=False, min_frequency=min_freq, 
                            special_tokens=list(special_tokens.values()), continuing_subword_prefix=prefix)
        tokenizer.train_from_iterator(corpus, trainer)
        tokenizer = self.enable(tokenizer, special_tokens)
        self.tokenizer = tokenizer
        self.trained = True
    
    def enable(self, tokenizer, special_tokens):

        """
        Enables the post processor of the huggingface Tokenizer.

        Args:
            tokenizer (tokenizers.Tokenizer): The instance of the huggingface Tokenizer.
            special_tokens (dict): A dictionary mapping special token names to their string values.

        Returns:
            tokenizers.Tokenizer: The huggingface Tokenizer with its post processor defined.
        """

        # enable post processor for tokenizer
        sos, eos = special_tokens["sos"], special_tokens["eos"]
        sos_id, eos_id = tokenizer.token_to_id(sos), tokenizer.token_to_id(eos)
        tokenizer.post_processor = processors.TemplateProcessing(single=f"{sos}:0 $A:0 {eos}:0", 
                                            pair=f"{sos}:0 $A:0 {eos}:0 $B:1 {eos}:1",
                                            special_tokens=[(f"{sos}", sos_id), (f"{eos}", eos_id)])
        return tokenizer
    
    def tokenize(self, text, special_tokens=True):

        """
        Tokenizes sequences of strings into tokens.

        Args:
            text (str or List[str]): An individual string or a list of strings.
            special_tokens (bool, optional): True includes special tokens in the tokenizer process, otherwise omitted. Default is True.

        Returns:
            List[List[str]]: A list containing the tokenized string or strings.
        """

        # tokenize sequences of text
        if isinstance(text, str):
            text = [text]
        encoded = self.tokenizer.encode_batch(text, add_special_tokens=special_tokens)
        return [encodings.tokens for encodings in encoded]
    
    def encode(self, text, special_tokens=True):

        """
        Encodes sequences of strings into tokens.

        Args:
            text (str or List[str]): An individual string or a list of strings.
            special_tokens (bool, optional): True includes special tokens in the tokenizer process. Default is True.

        Returns:
            List[List[int]]: A list containing the encoded string or strings.
        """

        # encode (to ids) sequences of text 
        if isinstance(text, str):
            text = [text]
        encoded = self.tokenizer.encode_batch(text, add_special_tokens=special_tokens)
        return [encoding.ids for encoding in encoded]
        
    def decode(self, ids, special_tokens=False):

        """
        Decodes sequences of ids into tokens.

        Args:
            ids (int or List[List[int]]): An individual integer or a list of integers.
            special_tokens (bool, optional): True includes special tokens in the tokenizer process. Default is True.

        Returns:
            List[str]: A list containing the decoded id sequence or ids sequences.
        """

        # decode encoded (ids) to strings
        if isinstance(ids, int):
            ids = [[ids]]
        decoded = self.tokenizer.decode_batch(ids, skip_special_tokens=not special_tokens)
        return decoded
    
    def pad(self, sequences, maxlen=None, end=True):

        """
        Pads sequences to a maximum length.

        Args:
            sequences (List[List[int]] or List[List[str]]): A list containing sequences of integers or strings.
            maxlen (int, optional): The maximum length to pad the sequence to. If not specificed, sequences will be padded to the longest sequence in the list. Default is None.
            end (bool, optional): True will add padding to the end and False adds padding to the start. Default is True.
            
        Returns:
            List[List[int]] or List[List[str]]: A list containing padded sequences of integers or strings.
        """

        # pad sequences to (possibly specified) maxlen
        maxlen = len(max(sequences, key=len)) if maxlen is None else maxlen
        return [self.pad_tokens(sequence, maxlen, end) for sequence in sequences]
        
    def pad_tokens(self, sequence, maxlen, end):

        """
        Pads a given sequence to a specified maximum length.

        Args:
            sequence (List[int] or List[str]): A sequence of integers or strings.
            maxlen (int): The maximum length to pad the sequence to.
            end: (bool): True will add padding to the end and False adds padding to the start.

        Returns:
            List[int] or List[str]: A padded sequence of integers or strings.
        """

        # pad a singular tokenized sequence
        tokenizer, special_tokens = self.tokenizer, self.special_tokens
        diff = maxlen - len(sequence) 
        assert(diff >= 0)
        pad = tokenizer.token_to_id(special_tokens["pad"]) if \
            isinstance(sequence[0], int) else special_tokens["pad"]
        padding = [pad for _ in range(diff)]
        return sequence + padding if end else padding + sequence
    
    def truncate(self, sequences, maxlen, end=True):

        """
        Truncates sequences based on a maximum length.

        Args:
            sequences (List[List[int]] or List[List[str]]): A list containing sequences of integers or strings.
            maxlen (int): The maximum length of a sequence.
            end (bool, optional): True will truncate at the end and False truncates at the start. Default is True.
            
        Returns:
            List[List[int]] or List[List[str]]: A list containing truncated sequences of integers or strings.
        """

        # truncate sequence to specified length
        return [self.truncate_tokens(sequence, maxlen, end) for sequence in sequences]

    def truncate_tokens(self, sequence, maxlen, end):

        """
        Truncates a given sequence based on a maximum length

        Args:
            sequence (List[int] or List[str]: A sequence of integers or strings.
            maxlen (int): The maximum length of the sequence.
            end: (bool): True will truncate at the end and False truncates at the start.

        Returns:
            List[int] or List[str]: A padded sequence of integers or strings.
        """

        # truncate a singular tokenized sequence
        return sequence[:maxlen] if end else sequence[-maxlen:]
    
    def __len__(self):

        """
        Gets the length of the huggingface Tokenzer

        Returns:
            int: The vocab size of the huggingface Tokenizer.
        """

        tokenizer = self.tokenizer
        return tokenizer.get_vocab_size()
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.tokenizer.token_to_id(item)
        if isinstance(item, int):
            return self.tokenizer.id_to_token(item)
        
class DualTokenizer:

    """
    DualTokenizer class for tokenizing, encoding, and decoding text for two languages.

    Args:
        source (utils.tokenizer.Tokenizer): The source language Tokenizer.
        target (utils.tokenizer.Tokenizer): The target language Tokenizer.

    Attributes:
        source (utils.tokenizer.Tokenizer): The Tokenizer for source language sequences.
        target (utils.tokenizer.Tokenizer): The Tokenizer for target language sequences.
    """   

    def __init__(self, source, target) -> None:
        self.source = source
        self.target = target

    def tokenize(self, text, special_tokens=True, module=None):

        """
        Tokenizes sequences of strings into tokens.

        Args:
            text (str or List[str]): An individual string or a list of strings.
            special_tokens (bool, optional): True includes special tokens in the tokenizer process, otherwise omitted. Default is True.

        Returns:
            List[List[str]]: A list containing the tokenized string or strings.
        """

        # tokenize text to source/target
        if module == "source":
            return self.source.tokenize(text, special_tokens)
        elif module == "target":
            return self.target.tokenize(text, special_tokens)
        raise ValueError(f"Module must be 'source' or 'target' not '{module}'.")

    def encode(self, text, special_tokens=True, module=None):

        """
        Encodes sequences of strings into tokens.

        Args:
            text (str or List[str]): An individual string or a list of strings.
            special_tokens (bool, optional): True includes special tokens in the tokenizer process. Default is True.

        Returns:
            List[List[int]]: A list containing the encoded string or strings.
        """

        # encode (to ids) text to source/target
        if module == "source":
            return self.source.encode(text, special_tokens)
        elif module == "target":
            return self.target.encode(text, special_tokens)
        raise ValueError(f"Module must be 'source' or 'target' not '{module}'.")

    def decode(self, ids, special_tokens=False, module=None):

        """
        Decodes encoded ids to text using either the source or target tokenizer.

        Args:
            ids (int or List[List[int]]): An individual integer or a list of integer.
            special_tokens (bool, optional): Whether to include special tokens in the decoding. Default is False.
            module (str, optional): Which tokenizer to use ('source' or 'target'). Default is None.

        Returns:
            List[str]: A list containing the decoded id sequence or ids sequences.
        """

        # decode enocded (ids) to source/target strings
        if module == "source":
            return self.source.decode(ids, special_tokens)
        elif module == "target":
            return self.target.decode(ids, special_tokens)
        raise ValueError(f"Module must be 'source' or 'target' not '{module}'.")
    
    def pad(self, sequences, maxlen=None, end=True):

        """
        Pads sequences to a maximum length.

        Args:
            sequences (List[List[int]] or List[List[str]]): A list containing sequences of integers or strings.
            maxlen (int, optional): The maximum length to pad the sequence to. If not specificed, sequences will be padded to the longest sequence in the list. Default is None.
            end (bool, optional): True will add padding to the end and False adds padding to the start. Default is True.
            
        Returns:
            List[List[int]] or List[List[str]]: A list containing padded sequences of integers or strings.
        """

        # pad sequences to (possibly specified) maxlen
        assert(self.source.tokenizer.token_to_id(self.source.special_tokens["pad"]) == \
               self.target.tokenizer.token_to_id(self.target.special_tokens["pad"]))
        return self.source.pad(sequences, maxlen, end)
        
    def truncate(self, sequences, maxlen, end=True):

        """
        Truncates sequences based on a maximum length.

        Args:
            sequences (List[List[int]] or List[List[str]]): A list containing sequences of integers or strings.
            maxlen (int): The maximum length of a sequence.
            end (bool, optional): True will truncate at the end and False truncates at the start. Default is True.
            
        Returns:
            List[List[int]] or List[List[str]]: A list containing truncated sequences of integers or strings.
        """

        # truncate a singular sequences sequence
        return self.source.truncate(sequences, maxlen, end)
    
    def vocab_size(self):

        """
        Returns the vocabulary size of both source and target tokenizers.

        Returns:
            Tuple[int, int]: A tuple containing the sizes of the source and target vocabularies respectively.
        """

        # return vocab of source & target tokenizers
        return len(self.source), len(self.target)
    
    def getitem(self, item, module=None):

        """
        Gets the corresponding token or ID from the source or target vocabulary.

        Args:
            item (str or int): The token (str) or id (int) to lookup.
            module (str, optional): Which tokenizer to use ('source' or 'target'). Default is None.

        Returns:
            str or int: If item is an id, returns the corresponding token; if item is a token, returns the corresponding id.
        """

        # get token/id of source/target
        if module == "source":
            return self.source[item]
        elif module == "target":
            return self.target[item]
        raise ValueError(f"Module must be 'source' or 'target' not '{module}'.")