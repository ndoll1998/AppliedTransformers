# import base dataset and tokenizer
from core.Dataset import BaseDataset
from transformers import PreTrainedTokenizer
# import utils
from core.utils import build_token_spans

class __RelationExtractionDatasetType(type):

    @property
    def num_relations(cls) -> int:
        return len(cls.RELATIONS)


class RelationExtractionDataset(BaseDataset, metaclass=__RelationExtractionDatasetType):
    """ Base Dataset for the Relation Extraction Task """

    RELATIONS = []

    def yield_item_features(self, train:bool, data_base_dir:str ='./data') -> iter:
        raise NotImplementedError()

    def build_dataset_item(self, text:str, entity_span_A:tuple, entity_span_B:tuple, label:str, seq_length:int, tokenizer:PreTrainedTokenizer):
        # get label id
        label = self.__class__.RELATIONS.index(label)
        # tokenize text and build token spans
        tokens = tokenizer.tokenize(text)[:seq_length]
        token_spans = build_token_spans(tokens, text)
        # find entity tokens
        entity_tokens_A = [i for i, (b, e) in enumerate(token_spans) if (entity_span_A[0] <= b) and (e <= entity_span_A[1])]
        entity_tokens_B = [i for i, (b, e) in enumerate(token_spans) if (entity_span_B[0] <= b) and (e <= entity_span_B[1])]
        # no overlap between entities
        if len(set(entity_tokens_A) & set(entity_tokens_B)) > 0:
            return None
        # at least one of both entities is out of bounds
        if (len(entity_tokens_A) == 0) or (len(entity_tokens_B) == 0):
            return None

        # find entity token spans
        entity_token_span_A = (entity_tokens_A[0], entity_tokens_A[-1] + 1)
        entity_token_span_B = (entity_tokens_B[0], entity_tokens_B[-1] + 1)
        # convert tokens to ids and pad to fill sequence length
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids += [tokenizer.pad_token_id] * (seq_length - len(token_ids))
        # return item
        return token_ids, entity_token_span_A, entity_token_span_B, label