# import base dataset and tokenizer
from core.Dataset import BaseDataset

class __AspectBasedSentimentAnalysisDatasetType(type):

    @property
    def num_labels(cls) -> int:
        return len(cls.LABELS)


class AspectBasedSentimentAnalysisDataset(BaseDataset, metaclass=__AspectBasedSentimentAnalysisDatasetType):
    """ Base Dataset for the aspect based sentiment analysis task """

    # list of all labels
    LABELS = []

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):
        raise NotImplementedError()

    def build_dataset_item(self, text:str, aspect_terms:list, labels:list, tokenizer):
            
        # tokenize text and aspect terms
        input_ids = tokenizer.encode(text)
        aspects_token_ids = [tokenizer.encode(term) for term in aspect_terms]

        # build labels
        if (labels is not None):
            # exactly one label per entity
            assert len(aspect_terms) == len(labels)
            # map labels to ids
            label2id = {l: i for i, l in enumerate(self.__class__.LABELS)}
            labels = [label2id[l] for l in labels]

        # return item
        return input_ids, aspects_token_ids, labels
