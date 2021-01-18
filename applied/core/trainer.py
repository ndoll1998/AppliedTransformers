import os
import torch
from tqdm import tqdm
# base model and dataset type
from .model import Model
from .dataset import Dataset
# import metric track
from .metrics import Track

class Trainer(object):
    """ Base class for trainers """

    # model and dataset types
    BASE_MODEL_TYPE:type = None
    BASE_DATASET_TYPE:type = None
    # metrics type
    METRIC_TYPE:type = None

    def __init__(self,
        model:Model,
        dataset:Dataset,
        optimizer:torch.optim.Optimizer
    ):
        # test types
        if not isinstance(model, self.__class__.BASE_MODEL_TYPE):
            raise TypeError("Model must inherit from %s! (%s)" % (
                self.__class__.BASE_MODEL_TYPE, model.__class__.__name__))
        if not isinstance(dataset, self.__class__.BASE_DATASET_TYPE):
            raise TypeError("Dataset must inherit from %s! (%s)" % (
                self.__class__.BASE_DATASET_TYPE, dataset.__class__.__name__))
        # save all
        self.model = model
        self.data = dataset
        self.optim = optimizer
        # prepare dataset
        self.data.prepare(self.model)
        # list to store al
        self.metrics = Track(self.__class__.METRIC_TYPE())

    def run_epoch(self) -> None:

        # train model
        self.model.train()
        train_running_loss = 0
        # progress bar
        with tqdm(total=len(self.data.train), ascii=True) as pbar:
            pbar.set_description("Train")

            for i, (x, labels) in enumerate(self.data.train, 1):
                # predict and compute
                logits = self.model.forward(*x)
                logits = (logits,) if isinstance(logits, torch.Tensor) else logits
                loss = self.model.loss(*logits, *labels)
                train_running_loss += loss.item()
                # backpropagate and update parameters
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # update progress bar
                pbar.set_postfix({'loss': train_running_loss / i})
                pbar.update(1)
                # TODO: remove for full training
                break

        # test model
        self.model.eval()
        eval_running_loss = 0
        all_logits, all_labels = None, None
        # no gradients needed for evaluation
        with torch.no_grad():
            # create progress bar
            with tqdm(total=len(self.data.eval), ascii=True) as pbar:
                pbar.set_description("Test")

                for i, (x, labels) in enumerate(self.data.eval, 1):
                    # evaluate batch and move all cached tensors to cpu
                    logits = self.model.forward(*x)
                    logits = (logits,) if isinstance(logits, torch.Tensor) else logits
                    loss = self.model.loss(*logits, *labels)
                    # update loss
                    eval_running_loss += loss.item()
                    # update all logits and labels
                    logits = [l.cpu() for l in logits]
                    labels = [l.cpu() for l in labels]
                    all_logits = logits if all_logits is None else [torch.cat(ll, dim=0) for ll in zip(all_logits, logits)]
                    all_labels = labels if all_labels is None else [torch.cat(ll, dim=0) for ll in zip(all_labels, labels)]
                    # update progress bar
                    pbar.set_postfix({'loss': eval_running_loss / i})
                    pbar.update(1)

                    # TODO: remove for full evaluation
                    break

        # add a new metrics entry
        self.metrics.add_entry(
            train_loss=train_running_loss / len(self.data.train), 
            eval_loss=eval_running_loss / len(self.data.eval),
            logits=[l.numpy() for l in all_logits],
            labels=[l.numpy() for l in all_labels]
        )
            
    def train(self, epochs:int, verbose:bool =True) -> None:
        for e in range(1, 1 + epochs):
            if verbose:
                print("Epoch %i" % e)
            self.run_epoch()
            if verbose:
                print("Metrics: %s" % str(self.metrics[-1]))
        return self
