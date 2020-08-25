# import torch and transformers
import torch
import transformers
# import base model
from .Model import BaseModel

class BaseDataset(torch.utils.data.TensorDataset):
    """ Base Class for Datasets """

    def __init__(self, train:bool, model:BaseModel, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str):
        
        # list of all dataset items
        data_items = [self.build_dataset_item(*feats, seq_length, tokenizer) for feats in self.yield_item_features(train, data_base_dir)]
        data_items = sum([model.prepare(*item, tokenizer=tokenizer) for item in data_items if item is not None], [])
        data_items = [item for item in data_items if item is not None]
        # separate into item features
        features = zip(*data_items)
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, *(torch.tensor(feat) for feat in features))

    def build_dataset_item(self, *feats, seq_length, tokenizer) -> tuple:
        raise NotImplementedError

    def yield_item_features(self, train:bool, data_base_dir:str):
        raise NotImplementedError