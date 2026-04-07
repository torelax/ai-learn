import os
import torch
import torch.optim as optim
from model.transformer import Transformer
from utils.checkpoint import Checkpoint
from utils.clock import Clock
from utils.dataset import Dataset
from utils.evaluator import Evaluator
from utils.parser import Parser
from utils.search import Beam, Greedy
from utils.test import prompt
from utils.tokenizer import DualTokenizer, Tokenizer
from utils.train import retrain, train
from utils.quantize import quantize_model
from utils.functional import save_config, save_module, save_model, load_module, load_model, graph, \
                                model_size, parameter_count, parse_config, read_data

precision_mapping = {
    "int8": torch.qint8,
    "float16": torch.float16
}

def main():

    args = Parser().parse_args()
    config_path = args.config
    verbose = args.verbose
    quantize = precision_mapping.get(args.quantize)
    search_type = args.search
    config = parse_config(config_path, verbose=verbose)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data configurations
    min_freq, maxlen, batch_size = config.get("min_freq"), config.get("maxlen"), config.get("batch_size")
    # model configurations
    dm, dk, dv, nhead, layers, dff = config.get("dm"), config.get("dk"), config.get("dv"), config.get("nhead"), config.get("layers"), config.get("dff")
    bias, dropout, eps, scale  = config.get("bias"), config.get("dropout"), config.get("eps"), config.get("scale")
    # optimizer & scheduler configurations
    lr, adam_eps, betas, weight_decay =  config.get("lr"), config.get("adam_eps"), config.get("betas"), config.get("weight_decay")
    factor, patience = config.get("factor"), config.get("patience")
    # decoder search configurations
    alpha, beam_width, search_eps, fast = config.get("alpha"), config.get("beam_width"), config.get("search_eps"), config.get("fast")
    # training & metric configurations
    eval_batch_size, goal_bleu, corpus_level, frequency, overwrite = config.get("eval_batch_size"), config.get("goal_bleu"), config.get("corpus_level"), config.get("frequency"), config.get("overwrite")
    warmups, epochs, clip = config.get("warmups"), config.get("epochs"), config.get("clip")

    if args.file == "train":

        # module paths
        checkpoint_path, config_save_path, dataloader_path, testloader_path = args.checkpoint, args.config_save_path, args.dataloader, args.testloader
        log_path, metrics_path, tokenizer_path, model_path = args.log, args.metrics, args.tokenizer, args.model

        # read data
        train_inputs_path, train_labels_path, test_inputs_path, test_labels_path = args.datasets
        train_inputs, train_labels, test_inputs, test_labels = read_data(train_inputs_path), read_data(train_labels_path), \
                                                                read_data(test_inputs_path), read_data(test_labels_path)
        # create tokenizer
        english, german = Tokenizer(), Tokenizer()
        english.train(train_inputs, min_freq=min_freq)
        german.train(train_labels, min_freq=min_freq)
        tokenizer = DualTokenizer(english, german)

        # create datasets & get corpus features
        trainset, testset = Dataset(train_inputs, train_labels, tokenizer), Dataset(test_inputs, test_labels, tokenizer)
        source_vocab, target_vocab = tokenizer.vocab_size()
        sos, eos, pad = tokenizer.getitem("<sos>", module="source"), tokenizer.getitem("<eos>", module="source"), \
                        tokenizer.getitem("<pad>", module="source")
        
        # create dataloader & testloader
        dataloader = trainset.dataloader(maxlen=maxlen, batch_size=batch_size, shuffle=False, drop_last=False)
        testloader = testset.dataloader(maxlen=maxlen, batch_size=eval_batch_size, shuffle=True, drop_last=False)

        # build modules
        model = Transformer(source_vocab, target_vocab, maxlen, pad_id=pad, dm=dm, dk=dk, dv=dv, nhead=nhead, 
                            layers=layers, dff=dff, bias=bias, dropout=dropout, eps=eps, scale=scale)
        model.to(device) # must be placed here
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=adam_eps)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
        checkpoint = Checkpoint(model, optimizer, scheduler, frequency=frequency, path=checkpoint_path, overwrite=overwrite)
        if search_type == "beam":
            search = Beam(sos, eos, maxlen, width=beam_width, alpha=alpha, eps=search_eps, fast=fast)
        elif search_type == "greedy":
            search = Greedy(sos, eos, maxlen, alpha=alpha, eps=search_eps)
        evaluator = Evaluator(testloader, tokenizer, search, goal_bleu=goal_bleu, corpus_level=corpus_level)
        clock = Clock()

        # save modules
        save_module(tokenizer, path=tokenizer_path, verbose=verbose)
        save_module(dataloader, path=dataloader_path, verbose=verbose)
        save_module(testloader, path=testloader_path, verbose=verbose)
        save_config(config_path=config_path, path=config_save_path, verbose=verbose)

        # display verbose (if applicable)
        if verbose:
            print(f"Using {device}")
            print(f"Number of input tokens: {source_vocab}\nNumber of output tokens: {target_vocab}")
            print(f"Average training sequence length: {trainset.avgseq()}\nLongest training sequence length: {trainset.maxseq()}")
            print(f"Average testing sequence length: {testset.avgseq()}\nLongest testing sequence length: {testset.maxseq()}")
            print(f"Maxlen: {maxlen}\nBatch Size: {batch_size}\nTest Batch Size: {eval_batch_size}")      
            print(f"Trainable Samples: {dataloader.size()}\nTestable Samples: {testloader.size()}")
            print(f"Number of Trainable Paramaters: {parameter_count(model):.1f}M\nSize of Model: {model_size(model):.1f}MB")

        # train model
        losses, test_losses, bleus = train(dataloader, model, optimizer, scheduler, evaluator, checkpoint, clock, epochs=epochs, 
                                           warmups=warmups, clip=clip, verbose=verbose, log=log_path, device=device)
        
        # graph metrics
        graph(losses, test_losses, bleus, path=metrics_path)

        # save trained model
        save_model(model, path=model_path, verbose=verbose)

        # save quantized model (if applicable)
        if quantize:
            dirname = os.path.dirname(model_path)
            model_path = os.path.join(dirname, f"model_{args.quantize}.pt")
            quantize_model(model, dtype=quantize, inplace=True)
            save_model(model, path=model_path, verbose=verbose)
        
    elif args.file == "retrain":

        # module paths
        checkpoint_path, dataloader_path, testloader_path = args.checkpoint, args.dataloader, args.testloader
        log_path, metrics_path, tokenizer_path, model_path = args.log, args.metrics, args.tokenizer, args.model

        # load dataloader testloader & tokenizer
        dataloader = load_module(path=dataloader_path, verbose=verbose)
        testloader = load_module(path=testloader_path, verbose=verbose)
        tokenizer = load_module(path=tokenizer_path, verbose=verbose)
        sos, eos, pad = tokenizer.getitem("<sos>", module="source"), tokenizer.getitem("<eos>", module="source"), \
                        tokenizer.getitem("<pad>", module="source")
        source_vocab, target_vocab = tokenizer.vocab_size()

        # build modules
        model = Transformer(source_vocab, target_vocab, maxlen, pad_id=pad, dm=dm, dk=dk, dv=dv, nhead=nhead, 
                            layers=layers, dff=dff,bias=bias, dropout=dropout, eps=eps, scale=scale)
        model.to(device) # must be placed here
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoint = Checkpoint(model, optimizer, scheduler)
        checkpoint.load_checkpoint(checkpoint_path, verbose=verbose, device=device)
        if search_type == "beam":
            search = Beam(sos, eos, maxlen, width=beam_width, alpha=alpha, eps=search_eps, fast=fast)
        elif search_type == "greedy":
            search = Greedy(sos, eos, maxlen, alpha=alpha, eps=search_eps)
        evaluator = Evaluator(testloader, tokenizer, search, goal_bleu=goal_bleu, corpus_level=corpus_level)
        clock = Clock(checkpoint["duration"])

        # display verbose
        if verbose:
            print(f"Using {device}")
            print(f"Number of input tokens: {source_vocab}\nNumber of output tokens: {target_vocab}")
            print(f"Maxlen: {maxlen}\nBatch Size: {batch_size}\nTest Batch Size: {eval_batch_size}")      
            print(f"Trainable Samples: {dataloader.size()}\nTestable Samples: {testloader.size()}")
            print(f"Number of Trainable Paramaters: {parameter_count(model):.1f}M\nSize of Model: {model_size(model):.1f}MB")

        # retrain model
        losses, test_losses, bleus = retrain(dataloader, checkpoint, evaluator, clock, epochs=epochs, warmups=warmups, 
                                             clip=clip, verbose=verbose, log=log_path, device=device)
        
        # graph metrics
        graph(losses, test_losses, bleus, path=metrics_path)

        # save trained model
        save_model(model, path=model_path, verbose=verbose)

        # save quantized model (if applicable)
        if quantize:
            dirname = os.path.dirname(model_path)
            model_path = os.path.join(dirname, f"model_{args.quantize}.pt")
            quantize_model(model, dtype=quantize, inplace=True)
            save_model(model, path=model_path, verbose=verbose)

    elif args.file == "prompt":

        # module paths
        tokenizer_path, model_path, early_stop = args.tokenizer, args.model, args.early_stop

        # load tokenizer & pull features
        tokenizer = load_module(path=tokenizer_path, verbose=verbose)
        sos, eos, pad = tokenizer.getitem("<sos>", module="source"), tokenizer.getitem("<eos>", module="source"), \
                        tokenizer.getitem("<pad>", module="source")
        en_vocab, de_vocab = tokenizer.vocab_size()

        # build model & load parameters
        model = Transformer(en_vocab, de_vocab, maxlen, pad_id=pad, dm=dm, dk=dk, dv=dv, nhead=nhead, 
                            layers=layers, dff=dff,bias=bias, dropout=dropout, eps=eps, scale=scale)
        load_model(model, path=model_path, verbose=verbose, device=device)

        # quantize (if applicable)
        if quantize:
            quantize_model(model, dtype=quantize, inplace=True)
            device = "cpu"

        # display verbose
        if verbose:
            print(f"Using {device}")
            print(f"Number of input tokens: {en_vocab}\nNumber of output tokens: {de_vocab}")
            print(f"Number of Trainable Paramaters: {parameter_count(model):.1f}M\nSize of Model: {model_size(model):.1f}MB")
            
        # create decoder search
        if search_type == "beam":
            search = Beam(sos, eos, maxlen, width=beam_width, alpha=alpha, eps=search_eps, fast=fast)
        elif search_type == "greedy":
            search = Greedy(sos, eos, maxlen, alpha=alpha, eps=search_eps)

        # prompt user w/ output
        model.to(device)
        sequence = input("Enter in the sequence of text: ")
        output = prompt(sequence, model, tokenizer, search, early_stop=early_stop, device=device)
        print(f"Translation: {output}")

if __name__ == "__main__":
    main()
