import os
import torch
from .model import Model
# utils
import itertools as it
from typing import Iterator
from tqdm import tqdm

__all__ = ['Dataset', 'DatasetItem']

class DatasetItem(object): pass
    
class Dataset(object):
    """ Base class for datasets """

    LABELS = []
    @classmethod
    def num_labels(cls) -> int:
        return len(cls.LABELS)

    def __init__(self, 
        data_base_dir:str ='./data', 
        seq_length:int =128,
        batch_size:int =64,
        device:str ='cpu'
    ) -> None:
        # arguments
        self.__base_dir = data_base_dir
        self.__seq_length = seq_length
        self.__batch_size = batch_size
        self.__device = device
        # training and testing datasets
        self.__train_data:torch.utils.data.Dataset = None
        self.__eval_data:torch.utils.data.Dataset = None
        # number of inputs and targets
        self.__n_inputs, self.__n_targets = None, None

    def to(self, device:str) -> "Dataset":
        self.__device = device
        return self

    @property
    def data_base_dir(self) -> str:
        return self.__base_dir
    @property
    def seq_length(self) -> int:
        return self.__seq_length
    @property
    def batch_size(self) -> int:
        return self.__batch_size
    @property
    def is_prepared(self) -> bool:
        return (self.__train_data is not None) and \
                (self.__eval_data is not None)

    @property
    def train(self) -> torch.utils.data.DataLoader:
        assert self.is_prepared
        return torch.utils.data.DataLoader(
            self.__train_data, 
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
    @property
    def eval(self) -> torch.utils.data.DataLoader:
        assert self.is_prepared
        return torch.utils.data.DataLoader(
            self.__eval_data, 
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )

    def yield_train_items(self) -> Iterator[DatasetItem]:
        raise NotImplementedError()
    def yield_eval_items(self) -> Iterator[DatasetItem]:
        raise NotImplementedError()

    def n_train_items(self) -> int:
        return None
    def n_eval_items(self) -> int:
        return None

    def _data_prep_pipe(self, 
        model:Model, 
        items:Iterator[DatasetItem], 
        n_items:int
    ) -> torch.utils.data.TensorDataset:
        # build all features
        all_features = []
        for item in tqdm(items, desc="Preparing", total=n_items, ascii=True):
            # 0) dataset -> yield items
            if (item is None): continue
            # 1) model -> build features
            features = model.build_features_from_item(item)
            if (features is None): continue
            # 2) feature -> tokenize text
            features = map(lambda f: f.tokenize(model.encoder.tokenizer), features)
            # 3) model -> truncate features
            truncate = lambda f: model.truncate_feature(f, max_seq_length=self.seq_length)
            features = map(truncate, features)
            features = filter(lambda f: f is not None, features)
            all_features.extend(features)
        # make sure the dataset is not empty
        assert len(all_features) > 0
        FeatureClass = all_features[0].__class__
        # 4) model/encoder -> build feature tensors
        input_features = model.encoder.build_feature_tensors(all_features)
        additional_features = FeatureClass.build_additional_feature_tensors(all_features)
        target_tensors = model.build_target_tensors(all_features)
        # get number of input and target tensors
        self.__n_inputs = len(input_features) + len(additional_features)
        self.__n_targets = len(target_tensors)
        # 5) build tensor dataset from feature tensors
        return torch.utils.data.TensorDataset(*input_features, *additional_features, *target_tensors)

    def prepare(self, model:Model, force:bool =False) -> "Dataset":
        # only prepare once
        if not self.is_prepared or force:
            # prepare training and testing dataset
            self.__train_data = self._data_prep_pipe(model, self.yield_train_items(), self.n_train_items())
            self.__eval_data = self._data_prep_pipe(model, self.yield_eval_items(), self.n_eval_items())
        # return self
        return self

    def statistics(self) -> str:
        header = "Split" + " " * 7 + "|"        
        train_split = "Training" + " " * 4 + "|"
        eval_split = "Evaluation" + " " * 2 + "|"
        # build the table column by column
        for i, label in enumerate(self.__class__.LABELS):
            # count the number of occurances
            # of the current label in both the
            # training and evaluation dataset
            # TODO: this only works for single label datasets
            #       what if target is a multi-dimensional tensor
            n_train = sum((t[0] == i).sum() for _, t in self.train)    
            n_eval = sum((t[0] == i).sum() for _, t in self.eval)    
            # update tables
            nc = len(label)
            header += " %s |" % label
            train_split += " %{0}i |".format(nc) % n_train
            eval_split += " %{0}i |".format(nc) % n_eval
        # combine rows of the table
        table = "\n".join([header, '-' * len(header), train_split, eval_split])
        return "\n" + self.__class__.__name__ + " Statistics:\n" + table + "\n"

    def __str__(self) -> str:
        return self.statistics()

    def _collate_fn(self, batch):
        tensors = torch.utils.data._utils.collate.default_collate(batch)
        tensors = tuple(t.to(self.__device) for t in tensors)
        return (tensors[:self.__n_inputs], tensors[self.__n_inputs:])
