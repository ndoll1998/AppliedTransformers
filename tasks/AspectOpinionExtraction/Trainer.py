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

    def predict_batch(self, *batch) -> tuple:
        # predict on batch
        outputs, (labels_a, labels_o) = self.model.preprocess_and_predict(*batch, tokenizer=self.tokenizer, device=self.device)
        loss, logits_a, logits_o = outputs[0], outputs[1], outputs[2]
        labels_a, labels_o = labels_a.to(self.device), labels_o.to(self.device)
        # get the valid labels and logits
        mask_a, mask_o = (labels_a >= 0), (labels_o >= 0)
        labels_a, logits_a = labels_a[mask_a], logits_a[mask_a, :]
        labels_o, logits_o = labels_o[mask_o], logits_o[mask_o, :]
        # return loss and cache for metrics
        return loss, (labels_a, labels_o, logits_a, logits_o)

    def compute_metrics(self, caches):
        # concatenate tensors from caches and get predictions from logits
        labels_a, labels_o, logits_a, logits_o = (torch.cat(l, dim=0) for l in zip(*caches))
        predicts_a, predicts_o = logits_a.max(dim=-1)[1], logits_o.max(dim=-1)[1]
        # compute f1-scores
        macro_f1_aspects = f1_score(predicts_a, labels_a, average='macro')
        macro_f1_opinions = f1_score(predicts_o, labels_o, average='macro')
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

