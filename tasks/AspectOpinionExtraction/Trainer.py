# import torch
import torch
# import base model and dataset
from .models import AspectOpinionExtractionModel
from .datasets import AspectOpinionExtractionDataset
# import base trainer and metrics
from core.Trainer import BaseTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt


class AspectOpinionExtractionTrainer(BaseTrainer):

    # base types
    BASE_MODEL_TYPE = AspectOpinionExtractionModel
    BASE_DATASET_TYPE = AspectOpinionExtractionDataset

    def compute_batch_loss(self, *batch):
        # preprocess batch
        kwargs, _, _ = self.model.preprocess(*batch, self.tokenizer, self.device)
        # apply model and get loss
        return self.model.forward(**kwargs)[0]

    def evaluate_batch(self, *batch):
        # preprocess batch and apply model
        kwargs, labels_a, labels_o = self.model.preprocess(*batch, self.tokenizer, self.device)
        loss, logits_a, logits_o = self.model.forward(**kwargs)[:3]
        # build targets
        mask = (labels_a != -1)
        target_a = labels_a[mask].cpu().tolist()
        target_o = labels_o[mask].cpu().tolist()
        # build predictions
        predicts_a = logits_a[mask, :].max(dim=-1)[1].cpu().tolist()
        predicts_o = logits_o[mask, :].max(dim=-1)[1].cpu().tolist()
        # return loss and cache
        return loss, (target_a, target_o, predicts_a, predicts_o)

    def compute_metrics(self, caches):
        # combine all caches
        targets_a, targets_o, predicts_a, predicts_o = (sum(l, []) for l in zip(*caches))
        # compute f1-scores
        macro_f1_aspects = f1_score(predicts_a, targets_a, average='macro')
        macro_f1_opinions = f1_score(predicts_o, targets_o, average='macro')
        # return metrics
        return macro_f1_aspects, macro_f1_opinions

    def plot(self, figsize=(8, 5)):
        # create figure
        fig, (loss_ax, f1_ax) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # plot train and test loss
        loss_ax.plot(self.metrics[0], label='Train')
        loss_ax.plot(self.metrics[1], label='Test')
        loss_ax.legend()
        loss_ax.set(xlabel='Epoch', ylabel='Loss')
        # plot f1-scores
        f1_ax.plot(self.metrics[2], label='Aspect')
        f1_ax.plot(self.metrics[3], label='Opinion')
        f1_ax.legend()
        f1_ax.set(xlabel='Epoch', ylabel='F1-Score')
        # return figure
        return fig

