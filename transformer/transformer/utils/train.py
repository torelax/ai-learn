import numpy as np
import torch.nn as nn
from utils.functional import write, printer, generate_masks, grad_norm

def train(dataloader, model, optimizer, scheduler=None, evaluator=None, checkpoint=None, clock=None, epochs=1000, warmups=100, clip=None, verbose=True, log=None, device=None):

    """
    Trains a given model on a dataset.

    Args:
        dataloader (utils.dataloader.DataLoader): The DataLoader object of the dataset.
        model (model.transformer.Transformer): The transformer model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Default is None.
        evaluator (utils.evaluator.Evaluator, optional): The evaluator to evaluate the model after each epoch. Default is None.
        checkpoint (utils.checkpoint.Checkpoint, optional): The checkpoint to save model states. Default is None.
        clock (utils.clock.Clock, optional): The clock to keep track of training time. Default is None.
        epochs (int, optional): The number of epochs to train. Default is 1000.
        warmups (int, optional): The number of warmup steps before reducing learning rate. Default is 100.
        clip (float, optional): The gradient clipping value. Default is None.
        verbose (bool, optional): Whether to print training progress. Default is True.
        log (str, optional): The path to a log file to write training progress. Default is None.
        device (torch.device, optional): The device to move tensors for computation. Default is None.

    Returns:
        Tuple[List[float], List[float], List[float]]: Lists of training losses, test losses, and BLEU scores.
    """

    # base values & modules
    m = len(dataloader)
    done = False
    cross_entropy = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    losses, test_losses, bleus = [], [], []
    clock_info, test_info, lr, warmup, saved = None, None, None, None, None

    # start clock
    if clock:
        clock.reset()
        clock.start()

    # show output (if applicable)
    output = f"{'-' * 70}\n|{f'Training Started'.center(68)}|\n{'-' * 70}"
    if verbose:
        print(output)

    # write output to log file (if applicanle)
    if log is not None:
        write(output, log, overwrite=True)

    for epoch in range(epochs):
        model.train() # set to train (may have been set to eval from evaluator)
        accum_loss = 0 # zero epoch loss 

        for batch in dataloader:
            # get src & trg tensors | shape: src - (batch_size, src_len) trg & out - (batch_size, trg_len)
            inputs, labels = batch.src, batch.trg
            src, trg, out = inputs, labels[:, :-1], labels[:, 1:]
            src, trg, out = src.long(), trg.long(), out.long()
            # generate the masks | shape: src_mask - (batch_size, src_len) trg_mask - (batch_size, trg_len, trg_len)
            src_mask, trg_mask = generate_masks(src, trg, model.pad_id)
            # move to device 
            src, trg, out = src.to(device), trg.to(device), out.to(device)
            src_mask, trg_mask = src_mask.to(device), trg_mask.to(device)

            # get pred, calc loss then backprop | shape: pred - (batch_size * seq_len, vocab_size) out - (batch_size * seq_len)
            optimizer.zero_grad()
            pred = model(src, trg, src_mask=src_mask, trg_mask=trg_mask) 
            pred, out = pred.contiguous().view(-1, pred.size(-1)), out.contiguous().view(-1)
            loss = cross_entropy(pred, out)
            loss.backward()

            # clip exploding grads (if applicable) & optimize
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            accum_loss += loss.item() # accumulate loss overtime

        # calc epoch loss, norm & retrieve current lr
        epoch_loss = accum_loss / m
        losses.append(epoch_loss)
        norm = grad_norm(model, p=2)
        lr = optimizer.param_groups[0]["lr"]

        # evaluate model (if applicable)
        if evaluator:
            bleu, test_loss = evaluator.evaluate(model, device=device)
            bleus.append(bleu)
            test_losses.append(test_loss)
            test_info = (test_loss, bleu)
            done = evaluator.done()

        # apply scheduler after warmups (if applicable)
        if scheduler:
            warmup = True
            if epoch + 1 > warmups:
                warmup = False
                # apply test loss (if applicable)
                if test_loss is not None:
                    scheduler.step(test_loss) 
                # otherwise apply train loss
                else:
                    scheduler.step(epoch_loss)

        # get times (if applicable)
        if clock:
            epoch_time, elapsed_time = clock.tick(), clock.elapsed()
            clock_info = epoch_time, elapsed_time

        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(losses=losses, test_losses=test_losses, bleus=bleus, duration=clock.duration)

        # show output (if applicable)
        output = printer(loss=epoch_loss, norm=norm, lr=lr, lr_round=3, epoch=epoch + 1, warmup=warmup, clock_info=clock_info, test_info=test_info, saved=saved, print_output=verbose)

        # write output to log file (if applicanle)
        if log:
            write(output, log, overwrite=False)

        # model meets bleu score (complete training)
        if done:
            break

    loss = np.mean(losses).item() # calc avg train loss
    # calc avg test loss & best bleu (if applicable)
    if test_info is not None:
        test_info = (np.mean(test_losses).item(), max(bleus))
    output = printer(loss=loss, clock_info=clock_info, test_info=test_info, print_output=verbose)
    
    # write output to log file (if applicanle)
    if log:
        write(output, log, overwrite=False)
    return losses, test_losses, bleus

def retrain(dataloader, checkpoint, evaluator=None, clock=None, epochs=1000, warmups=100, clip=None, verbose=True, log=None, device=None):

    """
    Continues training a transformer model from a checkpoint.

    Args:
        dataloader (utils.dataloader.DataLoader): The DataLoader object of the dataset.
        checkpoint (utils.checkpoint.Checkpoint): The checkpoint to load model states.
        evaluator (utils.evaluator.Evaluator, optional): The evaluator to evaluate the model after each epoch. Default is None.
        clock (utils.clock.Clock, optional): The clock to keep track of training time. Default is None.
        epochs (int, optional): The number of epochs to train. Default is 1000.
        warmups (int, optional): The number of warmup steps before reducing the learning rate. Default is 100.
        clip (float, optional): The gradient clipping value. Default is None.
        verbose (bool, optional): Whether to print training progress. Default is True.
        log (str, optional): The path to a log file to write training progress. Default is None.
        device (torch.device, optional): The device to move tensors for computation. Default is None.

    Returns:
        Tuple[List[float], List[float], List[float]]: Lists of training losses, test losses, and BLEU scores.
    """

    # grab attributes & modules from checkpoint
    model = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    scheduler = checkpoint["scheduler"]
    epoch_start = checkpoint["epoch"]
    losses = checkpoint["losses"]
    test_losses = checkpoint["test_losses"]
    bleus = checkpoint["bleus"]

    # base values & modules
    m = len(dataloader)
    done = False
    cross_entropy = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    clock_info, test_info, lr, warmup, saved = None, None, None, None, None

    # start clock (resume)
    if clock:
        clock.start()

    # show output (if applicable)
    output = f"{'-' * 70}\n|{f'Training Resumed'.center(68)}|\n{'-' * 70}"
    if verbose:
        print(output)

    # write output to log file (if applicanle)
    if log is not None:
        write(output, log, overwrite=False)

    for epoch in range(epoch_start, epochs):
        model.train() # set to train (may have been set to eval from evaluator)
        accum_loss = 0 # zero epoch loss

        for batch in dataloader:
            # get src & trg tensors | shape: src - (batch_size, src_len) trg & out - (batch_size, trg_len)
            inputs, labels = batch.src, batch.trg
            src, trg, out = inputs, labels[:, :-1], labels[:, 1:]
            src, trg, out = src.long(), trg.long(), out.long()
            # generate the masks | shape: src_mask - (batch_size, src_len) trg_mask - (batch_size, trg_len, trg_len)
            src_mask, trg_mask = generate_masks(src, trg, model.pad_id)
            # move to device 
            src, trg, out = src.to(device), trg.to(device), out.to(device)
            src_mask, trg_mask = src_mask.to(device), trg_mask.to(device)

            # get pred, calc loss then backprop | shape: pred - (batch_size * seq_len, vocab_size) out - (batch_size * seq_len)
            optimizer.zero_grad()
            pred = model(src, trg, src_mask=src_mask, trg_mask=trg_mask) 
            pred, out = pred.contiguous().view(-1, pred.size(-1)), out.contiguous().view(-1)
            loss = cross_entropy(pred, out)
            loss.backward()

            # clip exploding grads (if applicable) & optimize
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            accum_loss += loss.item() # accumulate loss overtime

        # calc epoch loss, norm & retrieve current lr
        epoch_loss = accum_loss / m
        losses.append(epoch_loss)
        norm = grad_norm(model, p=2)
        lr = optimizer.param_groups[0]["lr"]

        # evaluate model (if applicable)
        if evaluator:
            bleu, test_loss = evaluator.evaluate(model, device=device)
            bleus.append(bleu)
            test_losses.append(test_loss)
            test_info = (test_loss, bleu)
            done = evaluator.done()
            
        # apply scheduler after warmups (if applicable)
        if scheduler:
            warmup = True
            if epoch + 1 > warmups:
                warmup = False
                # apply test loss (if applicable)
                if test_loss is not None:
                    scheduler.step(test_loss) 
                # otherwise apply train loss
                else:
                    scheduler.step(epoch_loss)

        # get times (if applicable)
        if clock:
            epoch_time, elapsed_time = clock.tick(), clock.elapsed()
            clock_info = epoch_time, elapsed_time

        # check on checkpoint (if applicable)
        if checkpoint:
            saved = checkpoint.check(losses=losses, test_losses=test_losses, bleus=bleus, duration=clock.duration)

        # show output (if applicable)
        output = printer(loss=epoch_loss, norm=norm, lr=lr, lr_round=3, epoch=epoch + 1, warmup=warmup, clock_info=clock_info, test_info=test_info, saved=saved, print_output=verbose)

        # write output to log file (if applicanle)
        if log:
            write(output, log, overwrite=False)

        # model meets bleu score (complete training)
        if done:
            break

    loss = np.mean(losses).item() # calc avg train loss
    # calc avg test loss & best bleu (if applicable)
    if test_info is not None:
        test_info = (np.mean(test_losses).item(), max(bleus))
    output = printer(loss=loss, clock_info=clock_info, test_info=test_info, print_output=verbose)

    # write output to log file (if applicanle)
    if log:
        write(output, log, overwrite=False)
    return losses, test_losses, bleus