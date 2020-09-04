import torch
# import base models
from transformers import BertForSequenceClassification
from .AspectBasedSentimentAnalysisModel import AspectBasedSentimentAnalysisModel
# import utils
from core.utils import align_shape

class BertForSentencePairClassification(AspectBasedSentimentAnalysisModel, BertForSequenceClassification):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def build_feature_tensors(self, input_ids, aspects_token_ids, labels, seq_length=None, tokenizer=None) -> tuple:
        # one label per entity span
        assert (labels is None) or (len(aspects_token_ids) == len(labels))

        # add special tokens
        if input_ids[0] != tokenizer.cls_token_id:
            input_ids.insert(0, tokenizer.cls_token_id)
        if input_ids[-1] != tokenizer.sep_token_id:
            input_ids.append(tokenizer.sep_token_id)
        # remove special tokens from aspects tokens
        aspects_token_ids = [ids[1:] if ids[0] == tokenizer.cls_token_id else ids for ids in aspects_token_ids]
        aspects_token_ids = [ids[:-1] if ids[-1] == tokenizer.sep_token_id else ids for ids in aspects_token_ids]

        k = len(input_ids)
        # build overflow mask
        mask = [(seq_length is None) or (k + len(ids) + 1 <= seq_length) for ids in aspects_token_ids]
        # build token-type-ids and sentence pairs - ignore samples that would overflow the sequence length
        token_type_ids = [[0] * k + [1] * (len(ids) + 1) for ids, valid in zip(aspects_token_ids, mask) if valid]
        sentence_pairs = [input_ids + ids + [tokenizer.sep_token_id] for ids, valid in zip(aspects_token_ids, mask) if valid]
        # remove labels for examples that are out of bounds
        if labels is not None:
            labels = [l for l, valid in zip(labels, mask) if valid]

        # choose minimal sequence length to fit all examples
        if seq_length is None:
            seq_length = max((len(ids) for ids in sentence_pairs), default=0)
        # convert to tensors
        sentence_pairs = torch.LongTensor(align_shape(sentence_pairs, (len(sentence_pairs), seq_length), tokenizer.pad_token_id))
        token_type_ids = torch.LongTensor(align_shape(token_type_ids, (len(token_type_ids), seq_length), tokenizer.pad_token_id))
        labels = torch.LongTensor(labels) if labels is not None else None
        # return items
        return sentence_pairs, token_type_ids, labels

    def preprocess(self, input_ids, token_type_ids, labels, tokenizer) -> dict:
        # build masks
        attention_mask = (input_ids != tokenizer.pad_token_id)
        # build keyword arguments for forward call
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }, labels
        

""" KnowBert Model """

# import KnowBert Model and utils
from external.KnowBert.src.kb.model import KnowBertForSequenceClassification
from external.utils import knowbert_build_caches_from_input_ids
# import knowledge bases to register them
import external.KnowBert.src.knowledge

class KnowBertForSentencePairClassification(KnowBertForSequenceClassification, BertForSentencePairClassification):

    def build_feature_tensors(self, *args, tokenizer, **kwargs) -> tuple:
        # build feature tensors
        input_ids, token_type_ids, labels = BertForSentencePairClassification.build_feature_tensors(self, *args, tokenizer=tokenizer, **kwargs)
        # build caches
        caches = knowbert_build_caches_from_input_ids(self, input_ids, tokenizer)
        # return feature tensors and caches
        return (input_ids, token_type_ids, labels) + caches

    def preprocess(self, input_ids, token_type_ids, labels, *caches, tokenizer) -> dict:
        # set caches
        self.set_valid_kb_caches(*caches)
        # preprocess
        return BertForSentencePairClassification.preprocess(self, input_ids, token_type_ids, labels, tokenizer=tokenizer)