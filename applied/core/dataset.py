import torch
import itertools as it
from .model import Model
from typing import Iterator

__all__ = ['Dataset', 'DatasetItem']

class DatasetIter(object):
    def __init__(self, 
        dataloader:torch.utils.data.DataLoader, 
        device:str,
        split_at:int
    ):
        self.dataloader = dataloader
        self.device = device
        self.n = split_at
    def __len__(self) -> int:
        return len(self.dataloader)
    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        return self
    def __next__(self):
        tensors = next(self.dataloader_iter)
        return (
            tuple(t.to(self.device) for t in tensors[:self.n]),
            tuple(t.to(self.device) for t in tensors[self.n:])
        )

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
        # training and testing dataloaders
        self.__train_loader:torch.utils.data.DataLoader = None
        self.__eval_loader:torch.utils.data.DataLoader = None
        # number of inputs and targets
        self.__n_inputs, self.__n_targets = None, None

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
    def train(self) -> iter:
        return DatasetIter(
            dataloader=self.__train_loader,
            device=self.__device,
            split_at=self.__n_inputs
        )
    @property
    def eval(self) -> iter:
        return DatasetIter(
            dataloader=self.__eval_loader,
            device=self.__device,
            split_at=self.__n_inputs
        )

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
        features = list(map(truncate, features))
        # 4) model/encoder -> build feature tensors
        feature_tensors = model.encoder.build_feature_tensors(features)
        target_tensors = model.build_target_tensors(features)
        # get number of input and target tensors
        self.__n_inputs = len(feature_tensors)
        self.__n_targets = len(target_tensors)
        # 5) build tensor dataset from feature tensors
        return torch.utils.data.TensorDataset(*feature_tensors, *target_tensors)

    def prepare(self, model:Model) -> None:
        # prepare training and testing dataset
        train_dataset = self._data_prep_pipe(model, self.yield_train_items())
        eval_dataset = self._data_prep_pipe(model, self.yield_eval_items())
        # create dataloaders
        self.__train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.__eval_loader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=self.batch_size)
        # return self
        return self