# import torch
import torch
# import base dataset and tokenizer
from core.Dataset import BaseDataset
# import utils
from core.utils import build_token_spans

class __AspectBasedSentimentAnalysisDatasetType(type):

    @property
    def num_labels(cls) -> int:
        return len(cls.LABELS)


class AspectBasedSentimentAnalysisDataset(BaseDataset, metaclass=__AspectBasedSentimentAnalysisDatasetType):
    """ Base Dataset for the aspect based sentiment analysis task """

    MAX_ASPECT_TOKENS = 4
    MAX_ASPECT_NUMBER = 4
    # list of all labels
    LABELS = []

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):
        raise NotImplementedError()

    def build_dataset_item(self, text:str, aspect_terms:list, labels:list, seq_length:int, tokenizer):
        # exactly one label per entity
        assert len(aspect_terms) == len(labels)
        # build label-to-id map
        label2id = {l: i for i, l in enumerate(self.__class__.LABELS)}
        
        # tokenize text
        input_ids = tokenizer.encode(text)[:seq_length]
        input_ids += [tokenizer.pad_token_id] * (seq_length - len(input_ids))
        
        n, m = AspectBasedSentimentAnalysisDataset.MAX_ASPECT_NUMBER, AspectBasedSentimentAnalysisDataset.MAX_ASPECT_TOKENS
        # tokenize all aspect terms
        aspects_token_ids = [tokenizer.encode(term) for term in aspect_terms[:n]]
        aspects_token_ids = [ids + [tokenizer.pad_token_id] * (m - len(ids)) for ids in aspects_token_ids]
        aspects_token_ids += [[tokenizer.pad_token_id] * m] * (n - len(aspects_token_ids))
        # build labels
        labels = [label2id[l] for l in labels[:n]]
        labels = labels + [-1] * max(0, n - len(labels))

        # return item
        return input_ids, aspects_token_ids, labels
