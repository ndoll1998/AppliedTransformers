# import torch
import torch
import torch.nn as nn
# import utils
from dataclasses import dataclass
from typing import Union, Tuple, Dict

@dataclass
class FeaturePair(object):
    text:str
    labels:Union[Tuple[int], int]    # depending on task
    # optional
    tokens:Tuple[str] =None

    def tokenize(self, tokenizer) -> "FeaturePair":
        if self.tokens is None:
            self.tokens = tokenizer.tokenize(self.text)
        return self

    def token_ids(self, tokenizer) -> Tuple[int]:
        assert self.tokens is not None
        return tokenizer.convert_tokens_to_ids(self.tokens)

class Encoder(nn.Module):
    TOKENIZER_TYPE:type = None

    def __init__(self):
        # initialize module
        nn.Module.__init__(self)
        self.__tokenizer = None

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError()

    @property
    def tokenizer(self):
        return self.__tokenizer
    @tokenizer.setter
    def tokenizer(self, tokenizer):
        assert isinstance(tokenizer, self.__class__.TOKENIZER_TYPE)
        self.__tokenizer = tokenizer

    def init_tokenizer(self, *args, **kwargs):
        # initialize tokenizer
        assert self.__class__.TOKENIZER_TYPE is not None
        self.__tokenizer = self.__class__.TOKENIZER_TYPE(*args, **kwargs)

    def build_feature_tensors(self, features:tuple):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Model(nn.Module):
    
    def __init__(self, encoder:Encoder):
        nn.Module.__init__(self)
        # save base model
        self.__encoder = encoder

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def loss(self, *logits_and_labels):
        raise NotImplementedError()

    def build_features_from_item(self, item:'DatasetItem') -> Tuple[FeaturePair]:
        raise NotImplementedError()

    def build_target_tensors(self, features:Tuple[FeaturePair]) -> Tuple[torch.Tensor]:
        raise NotImplementedError()

    def truncate_feature(self, f:FeaturePair, max_seq_length:int) -> FeaturePair:
        """ This function may be overriden as needed for specific models or tasks 
            By default just cuts tokens that exceed the maximum sequence length.
        """
        f.tokens = f.tokens[:max_seq_length]
        f.labels = f.labels[:max_seq_length] if isinstance(f.labels, (tuple, list)) else f.labels
        return f

    def parameters(self, only_head:bool =False):
        return (p for n, p in nn.Module.named_parameters(self) 
                    if not only_head or ('__encoder' not in n))

