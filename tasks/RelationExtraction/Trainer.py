# import torch
import torch
# import model and tokenizer
from .models import RelationExtractionModel
# import datasets
from .datasets import RelationExtractionDataset
# import base trainer and metrics
from core.Trainer import BaseTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt


class RelationExtractionTrainer(BaseTrainer):

    BASE_MODEL_TYPE = RelationExtractionModel
    BASE_DATASET_TYPE = RelationExtractionDataset

    def __init__(self, 
        # model and tokenizer
        model_type:type =None,
        pretrained_name:str =None,
        model_kwargs:dict ={},
        device:str ='cpu',
        # data
        dataset_type:torch.utils.data.Dataset =None,
        data_base_dir:str ='./data',
        seq_length:int =None,
        batch_size:int =None,
        # optimizer
        learning_rate:float =None,
        weight_decay:float =None,
    ):
        # update model kwargs
        model_kwargs.update({'num_labels': dataset_type.num_relations})
        # initialize trainer
        BaseTrainer.__init__(self, 
            # model
            model_type=model_type,
            pretrained_name=pretrained_name,
            model_kwargs=model_kwargs,
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
        # update model embedding dimensions to match tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def compute_batch_loss(self, *batch):
        # convert dataset batch to inputs for model
        kwargs, _ = self.model.preprocess(*batch, self.tokenizer, self.device)
        # apply model and get loss
        return self.model.forward(**kwargs)[0]

    def evaluate_batch(self, *batch):
        # convert dataset batch to inputs for model and apply model
        kwargs, labels = self.model.preprocess(*batch, self.tokenizer, self.device)
        loss, logits = self.model.forward(**kwargs)[:2]
        # build targets and predictions
        targets = labels.cpu().tolist()
        predicts = logits.max(dim=-1)[1].cpu().tolist()
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

