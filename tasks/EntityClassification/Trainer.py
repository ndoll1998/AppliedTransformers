# import torch
import torch
# import base model, tokenizer and dataset
from .models import EntityClassificationModel
from .datasets import EntityClassificationDataset
# import base trainer and metrics
from core.Trainer import SimpleCrossEntropyTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt

class EntityClassificationTrainer(SimpleCrossEntropyTrainer):

    # set base types for model and dataset
    BASE_MODEL_TYPE = EntityClassificationModel
    BASE_DATASET_TYPE = EntityClassificationDataset

    def __init__(self, 
        # model and tokenizer
        model_type:type =None,
        pretrained_name:str =None,
        model_kwargs:dict ={},
        device:str ='cpu',
        # data
        dataset_type:torch.utils.data.Dataset =None,
        data_base_dir:str ='./data',
        dataset_kwargs:dict ={},
        seq_length:int =None,
        batch_size:int =None,
        # optimizer
        learning_rate:float =None,
        weight_decay:float =None,
    ):
        # update model kwargs
        model_kwargs.update({'num_labels': dataset_type.num_labels})
        # initialize trainer
        SimpleCrossEntropyTrainer.__init__(self, 
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
            dataset_kwargs=dataset_kwargs,
            seq_length=seq_length,
            batch_size=batch_size
        )
    