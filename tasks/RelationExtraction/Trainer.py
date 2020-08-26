# import torch
import torch
# import model and tokenizer
from .models import RelationExtractionModel
# import datasets
from .datasets import RelationExtractionDataset
# import base trainer and metrics
from core.Trainer import SimpleCrossEntropyTrainer
from sklearn.metrics import f1_score
# import matplotlib
from matplotlib import pyplot as plt

class RelationExtractionTrainer(SimpleCrossEntropyTrainer):

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
            seq_length=seq_length,
            batch_size=batch_size
        )
    
