import torch
# same model as in the aspect-based sentiment analysis task
from .EntityClassificationModel import EntityClassificationModel
from ...AspectBasedSentimentAnalysis.models import BertForSentencePairClassification as BaseModel
# import utils
from core.utils import align_shape

class BertForSentencePairClassification(EntityClassificationModel, BaseModel):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def prepare(self, input_ids, entity_spans, labels, seq_length=None, tokenizer=None) -> list:
        # one label per entity span
        assert (labels is None) or (len(entity_spans) == len(labels))

        # add special tokens
        if input_ids[0] != tokenizer.cls_token_id:
            input_ids.insert(0, tokenizer.cls_token_id)
            # update entity spans
            entity_spans = [(s+1, e+1) for s, e in entity_spans]
        if input_ids[-1] != tokenizer.sep_token_id:
            input_ids.append(tokenizer.sep_token_id)

        k = len(input_ids)
        # build overflow mask
        mask = [(seq_length is None) or (k + e - s + 1 <= seq_length) for s, e in entity_spans]
        # build token-type-ids and sentence pairs - ignore samples that would overflow the sequence length
        token_type_ids = [[0] * k + [1] * (e - s + 1) for (s, e), valid in zip(entity_spans, mask) if valid]
        sentence_pairs = [input_ids + input_ids[s:e] + [tokenizer.sep_token_id] for (s, e), valid in zip(entity_spans, mask) if valid]
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