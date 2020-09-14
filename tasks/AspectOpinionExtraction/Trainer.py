# import torch
import torch
# import base model and dataset
from .models import AspectOpinionExtractionModel
from .datasets import AspectOpinionExtractionDataset
# import base trainer and metrics
from core.Trainer import BaseTrainer
from sklearn.metrics import f1_score
# import utils
from core.utils import build_confusion_matrix, plot_confusion_matrix
from math import ceil
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
        # build confusion matrices
        aspect_confusion = build_confusion_matrix(labels_a, predicts_a)
        opinion_confusion = build_confusion_matrix(labels_o, predicts_o)
        # compute f1-scores
        macro_f1_aspects = f1_score(predicts_a, labels_a, average='macro')
        macro_f1_opinions = f1_score(predicts_o, labels_o, average='macro')
        # return metrics
        return macro_f1_aspects, macro_f1_opinions, aspect_confusion, opinion_confusion

    def metrics_string(self, metrics:tuple) -> str:
        return "Train Loss: %.4f\t- Test Loss: %.4f\t- Aspect F1: %.4f\t- Opinion F1: %.4f" % metrics[:4]

    def plot(self, figsize=(20, 15), **kwargs):
        # read metrics
        train_loss, test_loss, aspect_f1, opinion_f1, aspect_confusions, opinion_confusions = self.metrics
        # build layout and create figure
        n = 2 + ceil(len(aspect_confusions) / 6)
        fig = plt.figure(figsize=figsize, **kwargs)
        # plot train and test loss
        loss_ax = fig.add_subplot(n, 1, 1)
        loss_ax.plot(train_loss, label='Train')
        loss_ax.plot(test_loss, label='Test')
        loss_ax.legend()
        loss_ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
        # plot f1-scores
        f1_ax = fig.add_subplot(n, 1, 2)
        f1_ax.plot(aspect_f1, label='Aspect')
        f1_ax.plot(opinion_f1, label='Opinion')
        f1_ax.legend()
        f1_ax.set(xlabel='Epoch', ylabel='F1-Score', title='F1 Score')
        # plot all confusion matrices
        for i, (aspect_cm, opinion_cm) in enumerate(zip(aspect_confusions, opinion_confusions), 1):
            aspect_ax = fig.add_subplot(n, 6, 12 - 1 + i * 2)
            opinion_ax = fig.add_subplot(n, 6, 12 + i * 2)
            plot_confusion_matrix(aspect_ax, aspect_cm, title="Aspect - Epoch %i" % i)
            plot_confusion_matrix(opinion_ax, opinion_cm, title="Opinion - Epoch %i" % i)
        # set layout
        fig.tight_layout()
        # return figure
        return fig

