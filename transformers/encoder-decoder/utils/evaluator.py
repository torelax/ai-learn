from utils.test import test, predict
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

class Evaluator:

    """
    A class used to evaluate a model based on loss and BLEU score.

    Args:
        dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the data.
        search (utils.search.DecoderSearch): The search algorithm used for prediction.
        goal_bleu (float, optional): The goal BLEU score to reach. Default is 25.
        corpus_level (bool, optional): How to compute bleu score over dataloader. False computes bleu score on a sentence level. Default is False.

    Attributes:
        dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
        tokenizer (utils.tokenizer.Tokenizer): The tokenizer used to tokenize the data.
        search (utils.search.DecoderSearch): The search algorithm used for prediction.
        goal_bleu (float): The goal BLEU score to reach.
        corpus_level (bool): Method to compute bleu scores over dataloader.
        bleu (float): The current BLEU score.
        loss (float): The current loss.
    """

    def __init__(self, dataloader, tokenizer, search, goal_bleu=25, corpus_level=False):
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.search = search
        self.goal_bleu = goal_bleu
        self.corpus_level = corpus_level
        self.bleu = 0
        self.loss = float("inf")

    def evaluate(self, model, device=None):

        """
        Evaluates and updates the BLEU score and loss of the model.

        Args:
            model (model.transformer.Transformer): The transformer model to be evaluated.
            device (torch.device, optional): The device to move tensors for computation. Defaults to None.

        Returns:
            Tuple[float, float]: The BLEU score and loss of the model.
        """

        # calculate & update both best loss & best bleu
        bleu, loss = self.evaluate_bleu(model, self.dataloader, device=device), self.evaluate_loss(model, self.dataloader, device=device)
        self.bleu, self.loss = max(self.bleu, bleu), min(self.loss, loss)
        return bleu, loss
    
    def evaluate_loss(self, model, dataloader, device=None):

        """
        Evaluates the loss of the model on a DataLoader.

        Args:
            model (model.transformer.Transformer): The transformer model to be evaluated.
            dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
            device (torch.device, optional): The device to move tensors for computation. Defaults to None.

        Returns:
            float: The loss of the model.
        """

        # evaluate loss from dataloader
        loss = test(dataloader, model, device=device)
        return loss
    
    def evaluate_bleu(self, model, dataloader, device=None):

        """
        Evaluates the BLEU score of the model on a DataLoader.

        Args:
            model (model.transformer.Transformer): The transformer model to be evaluated.
            dataloader (utils.dataloader.DataLoader): The DataLoader used to iterate over the data.
            device (torch.device, optional): The device to move tensors for computation. Defaults to None.

        Returns:
            float: The BLEU score of the model.
        """

        tokenizer, search, corpus_level = self.tokenizer, self.search, self.corpus_level
        
        # get predictions & calculate bleu score for batches
        predictions, m, bleu = predict(dataloader, model, search, device=device), len(dataloader), 0
        for hypothesis, references in predictions:
            hypothesis = tokenizer.decode(hypothesis.tolist(), special_tokens=False, module="target")
            references = tokenizer.decode(references.tolist(), special_tokens=False, module="target")
            hypothesis = tokenizer.tokenize(hypothesis, special_tokens=False, module="target")
            references = tokenizer.tokenize(references, special_tokens=False, module="target")

            # compute corpus level for each batch (bleu according to overall batch score)
            if corpus_level:
                references = [[ref] for ref in references]
                score = corpus_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method1)
                bleu += score * 100
            # compute sentence level for each batch (bleu according to ach sentence in the batch)
            else:
                batch_size, batch_bleu = len(hypothesis), 0
                for hyp, ref in zip(hypothesis, references):
                    score = sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1)
                    batch_bleu += score * 100
                # accumulate average bleu score for batch
                bleu += batch_bleu / batch_size

        # return average bleu score for dataloader
        return bleu / m
            
    def done(self):

        """
        Checks if the goal BLEU score has been reached.

        Returns:
            bool: True if the goal BLEU score has been reached, False otherwise.
        """

        # indicate if model metrics exceed goal
        return self.bleu >= self.goal_bleu