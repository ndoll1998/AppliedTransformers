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
    tokens:Tuple[int] =None
    
    def get_token_ids(self, tokenizer) -> Tuple[str]:
        # check if tokens are given
        if self.tokens is None:
            return tokenizer.encode(self.text)
        # if so use them
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

    def build_feature_tensors(self, features:tuple, seq_length:int):
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

    def preprocess(self, *batch) -> Tuple[dict, torch.tensor]:
        """ Preprocess batch and return input kwargs to forward pass and labels """
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def loss(self, *logits_and_labels):
        raise NotImplementedError()

    def build_features_from_item(self, item:'DatasetItem') -> Tuple[FeaturePair]:
        raise NotImplementedError()

    def build_target_tensors(self, features:Tuple[FeaturePair], seq_length:int):
        raise NotImplementedError()

    def parameters(self, only_head:bool =False):
        return (p for n, p in nn.Module.named_parameters(self) 
                    if not only_head or ('__encoder' not in n))

