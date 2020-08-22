# import torch
import torch
# import model and tokenizer
from .modeling import BertForAspectOpinionExtraction
from transformers import BertTokenizer
# import datasets
from .datasets import AspectOpinionExtractionDataset
# import base trainer and metrics
from base import BaseTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt


class AspectOpinionExtractionTrainer(BaseTrainer):

    # dataset type
    BASE_DATASET_TYPE = AspectOpinionExtractionDataset

    def __init__(self, 
        # model
        bert_base_model:str,
        device:str,
        # dataset parameters
        dataset_type:type,
        data_base_dir:str,
        seq_length:int,
        batch_size:int
        # optimizer
        learning_rate:float,
        weight_decay:float,
    ):
        # initialize trainer
        BaseTrainer.__init__(self, 
            # model
            bert_model_type=BertForAspectOpinionExtraction,
            tokenizer_type=BertTokenizer,
            bert_base_model=bert_base_model,
            device=device,
            # optimizer
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            # data
            dataset_type=dataset_type,
            data_base_dir=data_base_dir,
            seq_length=seq_length,
            batch_size=batch_size
        )

    def compute_batch_loss(self, input_ids, labels_a, labels_o):
        # move all to device and build attention mask
        input_ids, labels_a, labels_o = input_ids.to(self.device), labels_a.to(self.device), labels_o.to(self.device)
        mask = (input_ids != self.tokenizer.pad_token_id)
        # apply model and get loss
        return self.model.forward(
            input_ids, attention_mask=mask,
            aspect_labels=labels_a, opinion_labels=labels_o
        )[0]

    def evaluate_batch(self, input_ids, labels_a, labels_o):
        # move all to device and build attention mask
        input_ids, labels_a, labels_o = input_ids.to(self.device), labels_a.to(self.device), labels_o.to(self.device)
        mask = (input_ids != self.tokenizer.pad_token_id)
        # apply model and get loss
        loss, logits_a, logits_o = self.model.forward(
            input_ids, attention_mask=mask,
            aspect_labels=labels_a, opinion_labels=labels_o
        )[:3]
        # build targets
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

