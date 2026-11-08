import argparse

class Parser:

    """
    A class used to parse arguments from the command line for the Transformer Terminal Interface.

    Attributes:
        parser (argpase.ArgumentParser): The ArgumentParser used to parse command line arguments.
    """

    def __init__(self):

        self.parser = self.create_parser()

    def create_parser(self):

        """
        Creates an ArgumentParser with the necessary arguments.

        Returns:
            argparse.ArgumentParser: The created ArgumentParser.
        """

        parser = argparse.ArgumentParser(prog="Transformer Terminal Interface", 
                    description="Train, retrain, or prompt transformer model w/ optional arguments")
        subparser = parser.add_subparsers(dest="file")
        parser.add_argument("--config", type=str,
                            help="The path to the configurations file (e.g. '--config path/to/config.txt'), default: 'config.txt'",
                            default="config.txt")
        parser.add_argument("--model", type=str,
                            help="The path to save/load a model's parameters (e.g. '--model path/to/model.pt'), default: experiment/model.pt",
                            default="experiment/model.pt")   
        parser.add_argument("--search", type=str, choices=["beam", "greedy"],
                            help="The type of search to use (e.g. '--search greedy'), choices: 'greedy' & 'beam', default: 'beam'",
                            default="beam")     
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="Choose to display or not display verbose (e.g. '--verbose' [displays verbose]), default: True",
                            default=False)
        parser.add_argument("--quantize", type=str, choices=["int8", "float16"],
                            help="Choose to save/use a dynamically quantized model (e.g. '--quantize 8'), choices: 'int8' & 'float16', default: None",
                            default=None)
        self.prompt_parser(subparser)
        self.retrain_parser(subparser)
        self.train_parser(subparser)
        return parser
    
    def parse_args(self):

        """
        Parses the arguments from the command line.

        Returns:
            argparse.Namespace: The parsed arguments from the command line.
        """

        return self.parser.parse_args()

    def prompt_parser(self, subparser):

        """
        Adds specific arguments for the 'prompt' command to a subparser.

        Args:
            subparser (argparse._SubParsersAction): The subparser to which the arguments are added.
        """

        prompt_parser = subparser.add_parser("prompt")
        prompt_parser.add_argument("--tokenizer", type=str,
                            help="The path to load the tokenizer (e.g. '--tokenizer path/to/tokenizer.pt'), default: 'experiment/tokenizer.pt'",
                            default="experiment/tokenizer.pt")
        prompt_parser.add_argument("--early-stop", action="store_true",
                            help="To use early stopping when prompting (e.g. '--early-stop [passes early_stop as True during prompting]), default: False",
                            default=False)

    def retrain_parser(self, subparser):

        """
        Adds specific arguments for the 'retrain' command to a subparser.

        Args:
            subparser (argparse._SubParsersAction): The subparser to which the arguments are added.
        """

        retrain_parser = subparser.add_parser("retrain")
        retrain_parser.add_argument("--checkpoint", type=str,
                            help="The path to load the module's checkpoints (e.g. '--checkpoint path/to/checkpoint-###.pt'), default: None",
                            default=None,
                            required=True)
        retrain_parser.add_argument("--dataloader", type=str,
                            help="The path to load the dataloader (e.g. '--dataloader path/to/dataloader.pt'), default: 'experiment/dataloader.pt'",
                            default="experiment/dataloader.pt")
        retrain_parser.add_argument("--log", type=str,
                            help="The path to load & update the log file w/ training info (e.g. '--log path/to/log.txt'), default: 'experiment/log.txt'",
                            default="experiment/log.txt")
        retrain_parser.add_argument("--metrics", type=str,
                            help="The path to load the metrics graph of training (e.g. '--metrics path/to/metrics.jpg'), default: 'experiment/metrics.jpg'",
                            default="experiment/metrics.jpg")        
        retrain_parser.add_argument("--tokenizer", type=str,
                            help="The path to load the tokenizer (e.g. '--tokenizer path/to/tokenizer.pt'), default: 'experiment/tokenizer.pt'",
                            default="experiment/tokenizer.pt")
        retrain_parser.add_argument("--testloader", type=str,
                            help="The path to load the testloader (e.g. '--testloader path/to/testloader.pt'), default: 'experiment/testloader.pt'",
                            default="experiment/testloader.pt")
        
    def train_parser(self, subparser):

        """
        Adds specific arguments for the 'train' command to a subparser.

        Args:
            subparser (argparse._SubParsersAction): The subparser to which the arguments are added.
        """

        train_parser = subparser.add_parser("train")
        train_parser.add_argument("--checkpoint", type=str,
                            help="The path to save the module's checkpoints (e.g. '--checkpoint path/to/checkpoint.pt'), default: 'experiment/checkpoint.pt'",
                            default="experiment/checkpoint.pt")
        train_parser.add_argument("--config-save-path", type=str,
                            help="The path to save the config.py file (e.g. '--config-save-path path/to/config.txt'), default: 'experiment/config.txt'",
                            default="experiment/config.txt")
        train_parser.add_argument("--dataloader", type=str,
                            help="The path to save the dataloader (e.g. '--dataloader path/to/dataloader.pt'), default: 'experiment/dataloader.pt'",
                            default="experiment/dataloader.pt")
        train_parser.add_argument("--datasets", type=str, nargs=4, 
                            help=" ".join("The path to each dataset for both training & testing separated by a space \
                            (e.g. '--datasets datasets/dataset1.txt dataset2.txt ...'), \
                            default: 'datasets/multi30k-train.en datasets/multi30k-train.de datasets/multi30k-test.en datasets/multi30k-test.de'".split()),
                            default=["datasets/multi30k-train.en", "datasets/multi30k-train.de", "datasets/multi30k-test.en", "datasets/multi30k-test.de"])
        train_parser.add_argument("--log", type=str,
                            help="The path to save & update the log file w/ training info (e.g. '--log path/to/log.txt'), default: 'experiment/log.txt'",
                            default="experiment/log.txt")
        train_parser.add_argument("--metrics", type=str,
                            help="The path to save the metrics graph of training (e.g. '--metrics path/to/metrics.jpg'), default: 'experiment/metrics.jpg'",
                            default="experiment/metrics.jpg")
        train_parser.add_argument("--tokenizer", type=str,
                            help="The path to save the tokenizer (e.g. '--tokenizer path/to/tokenizer.pt'), default: 'experiment/tokenizer.pt'",
                            default="experiment/tokenizer.pt")
        train_parser.add_argument("--testloader", type=str,
                            help="The path to save the testloader (e.g. '--testloader path/to/testloader.pt.pt'), default: 'experiment/testloader.pt'",
                            default="experiment/testloader.pt")