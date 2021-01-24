import os
import torch
from .model import Model
# utils
import itertools as it
from typing import Iterator
from applied.common.utils import fetch

__all__ = ['Dataset', 'DatasetItem']

class DatasetItem(object): pass
    
class Dataset(object):
    """ Base class for datasets """

    LABELS = []
    @classmethod
    def num_labels(cls) -> int:
        return len(cls.LABELS)

    # map data files to download urls
    # this enables auto downloading of the dataset
    AUTO_DOWNLOAD_AVAILABLE = False
    URL_FILE_MAP = {}

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
        # training and testing dataloaders
        self.__train_loader:torch.utils.data.DataLoader = None
        self.__eval_loader:torch.utils.data.DataLoader = None
        # number of inputs and targets
        self.__n_inputs, self.__n_targets = None, None
        # download dataset if needed
        self.download()

    def download(self) -> None:
        if not self.__class__.AUTO_DOWNLOAD_AVAILABLE:
            return
        for fpath, url in self.__class__.URL_FILE_MAP.items():
            # check if file already exists
            full_fpath = os.path.join(self.data_base_dir, fpath)
            if os.path.isfile(full_fpath): continue
            # create directory
            fpath, fname = os.path.split(full_fpath)
            os.makedirs(fpath, exist_ok=True)
            # fetch file
            fetch(url, save_to=full_fpath)

    def to(self, device:str) -> None:
        self.__device = device

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
    def train(self) -> torch.utils.data.DataLoader:
        return self.__train_loader
    @property
    def eval(self) -> torch.utils.data.DataLoader:
        return self.__eval_loader

    def yield_train_items(self) -> Iterator[DatasetItem]:
        raise NotImplementedError()
    def yield_eval_items(self) -> Iterator[DatasetItem]:
        raise NotImplementedError()

    def _data_prep_pipe(self, model:Model, items:Iterator[DatasetItem]) -> torch.utils.data.TensorDataset:
        # 0) dataset -> yield items
        drop_none = lambda f: f is not None
        items = filter(drop_none, items)
        # 1) model -> build features
        features = map(model.build_features_from_item, items)
        features = filter(drop_none, features)
        features = it.chain(*features)
        # 2) feature pair -> tokenize text
        features = map(lambda f: f.tokenize(model.encoder.tokenizer), features)
        # 3) model -> truncate features
        truncate = lambda f: model.truncate_feature(f, max_seq_length=self.seq_length)
        features = map(truncate, features)
        features = filter(drop_none, features)
        features = list(features)
        # 4) model/encoder -> build feature tensors
        input_features = model.encoder.build_feature_tensors(features)
        additional_features = features[0].__class__.build_additional_feature_tensors(features)
        target_tensors = model.build_target_tensors(features)
        # get number of input and target tensors
        self.__n_inputs = len(input_features) + len(additional_features)
        self.__n_targets = len(target_tensors)
        # 5) build tensor dataset from feature tensors
        return torch.utils.data.TensorDataset(*input_features, *additional_features, *target_tensors)

    def prepare(self, model:Model) -> None:
        # prepare training and testing dataset
        train_dataset = self._data_prep_pipe(model, self.yield_train_items())
        eval_dataset = self._data_prep_pipe(model, self.yield_eval_items())
        # create dataloaders
        self.__train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self._collate_fn)
        self.__eval_loader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self._collate_fn)
        # return self
        return self

    def _collate_fn(self, batch):
        tensors = torch.utils.data._utils.collate.default_collate(batch)
        tensors = tuple(t.to(self.__device) for t in tensors)
        return (tensors[:self.__n_inputs], tensors[self.__n_inputs:])