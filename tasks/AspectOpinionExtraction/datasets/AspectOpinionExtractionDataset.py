# import base dataset and tokenizer
from core.Dataset import BaseDataset
from transformers import BertTokenizer
# import utils
from core.utils import build_token_spans, mark_bio_scheme

class AspectOpinionExtractionDataset(BaseDataset):
    """ Base Dataset for the Aspect-Opinion Extraction Task """

    def yield_item_features(self, train:bool, base_data_dir:str):
        raise NotImplementedError()

    @classmethod
    def build_dataset_item(cls, text:str, aspect_spans:list =None, opinion_spans:list =None, tokenizer:BertTokenizer =None):
        # tokenize text and build token spans
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_spans = build_token_spans(tokens, text)
        # create bio schemes for aspects and opinions
        aspect_bio = mark_bio_scheme(token_spans, aspect_spans) if aspect_spans is not None else None
        opinion_bio = mark_bio_scheme(token_spans, opinion_spans) if opinion_spans is not None else None
        # return item
        return token_ids, aspect_bio, opinion_bio
