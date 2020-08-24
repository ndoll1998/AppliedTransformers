# import pytorch
import torch
# import base dataset
from core.Dataset import BaseDataset
# import tokenizer
from transformers import BertTokenizer
# import utils
from core.utils import build_token_spans, mark_bio_scheme

class AspectOpinionExtractionDataset(BaseDataset):
    """ Base Dataset for the Aspect-Opinion Extraction Task """

    def yield_item_features(self, train:bool, base_data_dir:str):
        raise NotImplementedError()

    def build_dataset_item(self, text:str, aspect_spans:list, opinion_spans:list, seq_length:int, tokenizer:BertTokenizer):
        # tokenize text and build token spans
        tokens = tokenizer.tokenize(text)[:seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_spans = build_token_spans(tokens, text)
        # create bio schemes for aspects and opinions
        aspect_bio = mark_bio_scheme(token_spans, aspect_spans)
        opinion_bio = mark_bio_scheme(token_spans, opinion_spans)
        # fill sequence length
        token_ids += [tokenizer.pad_token_id] * (seq_length - len(token_ids))
        aspect_bio += [-1] * (seq_length - len(aspect_bio))
        opinion_bio += [-1] * (seq_length - len(opinion_bio))
        # return item
        return token_ids, aspect_bio, opinion_bio
