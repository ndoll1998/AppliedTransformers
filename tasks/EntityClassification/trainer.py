# import torch and transformers
import torch
import transformers
# import model and tokenizer
from .modeling import BertForEntityClassification
from .tokenization import BertForEntityLinkingTokenizer
# import datasets
from .datasets import EntityClassificationDataset
# import base trainer and metrics
from base import BaseTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt


class EntityClassificationTrainer(BaseTrainer):

    BASE_DATASET_TYPE = EntityClassificationDataset

    def __init__(self, 
        # model
        bert_base_model:str,
        device:str,
        # data
        dataset_type:type,
        data_base_dir:str,
        seq_length:int,
        batch_size:int,
        # optimizer
        learning_rate:float,
        weight_decay:float,
    ):
        # initialize trainer
        BaseTrainer.__init__(self, 
            # model
            bert_model_type=None,   # manually initialize model
            tokenizer_type=BertForEntityLinkingTokenizer,
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
        # create model
        self.model = BertForEntityClassification.from_pretrained(bert_base_model, num_labels=self.train_dataloader.dataset.num_labels)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        # create optimizer
        self.optim = transformers.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
    
    def compute_batch_loss(self, input_ids, entity_starts, labels):
        # move all to device and build masks
        input_ids, entity_starts, labels = input_ids.to(self.device), entity_starts.to(self.device), labels.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        entity_mask = (labels != -1)
        # apply model and get loss
        return self.model.forward(
            input_ids, 
            attention_mask=attention_mask, 
            entity_starts=entity_starts, 
            entity_mask=entity_mask, 
            labels=labels
        )[0]

    def evaluate_batch(self, input_ids, entity_starts, labels):
        # move all to device and build masks
        input_ids, entity_starts, labels = input_ids.to(self.device), entity_starts.to(self.device), labels.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        entity_mask = (labels != -1)
        # apply model and get loss
        loss, logits = self.model.forward(
            input_ids, 
            attention_mask=attention_mask, 
            entity_starts=entity_starts, 
            entity_mask=entity_mask, 
            labels=labels
        )[:2]
        # build targets and predictions
        targets = labels[entity_mask].cpu().tolist()
        predicts = logits[entity_mask].max(dim=-1)[1].cpu().tolist()
        # return loss and cache
        return loss, (targets, predicts)

    def compute_metrics(self, caches):
        # combine all caches
        targets, predicts = (sum(l, []) for l in zip(*caches))
        # compute f1-scores
        macro_f1 = f1_score(predicts, targets, average='macro')
        micro_f1 = f1_score(predicts, targets, average='micro')
        # return metrics
        return macro_f1, micro_f1

    def plot(self, figsize=(8, 5)):
        # create figure
        fig, (loss_ax, f1_ax) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # plot train and test loss
        loss_ax.plot(self.metrics[0], label='Train')
        loss_ax.plot(self.metrics[1], label='Test')
        loss_ax.legend()
        loss_ax.set(xlabel='Epoch', ylabel='Loss')
        # plot f1-scores
        f1_ax.plot(self.metrics[2], label='Macro')
        f1_ax.plot(self.metrics[3], label='Micro')
        f1_ax.legend()
        f1_ax.set(xlabel='Epoch', ylabel='F1-Score')
        # return figure
        return fig

