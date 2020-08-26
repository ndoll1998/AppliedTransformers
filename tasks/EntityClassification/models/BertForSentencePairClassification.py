# import base models
from transformers import BertForSequenceClassification
from .EntityClassificationModel import EntityClassificationModel

class BertForSentencePairClassification(EntityClassificationModel, BertForSequenceClassification):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def prepare(self, input_ids, entity_spans, labels, tokenizer) -> list:

        n = len(input_ids)
        # remove padding tokens, entity spans and labels
        input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
        entity_spans = [(s, e) for s, e in entity_spans if s < e]
        labels = [l for l in labels if l != -1]
        assert len(entity_spans) == len(labels)

        # add special tokens
        if input_ids[0] != tokenizer.cls_token_id:
            input_ids.insert(0, tokenizer.cls_token_id)
            # update entity spans
            entity_spans = [(s+1, e+1) for s, e in entity_spans]
        if input_ids[-1] != tokenizer.sep_token_id:
            input_ids.append(tokenizer.sep_token_id)

        k = len(input_ids)
        # build token-type-ids - ignore samples that would overflow the sequence length
        token_type_ids = [[0] * k + [1] * (e - s + 1) for s, e in entity_spans if k + e - s + 1 <= n]
        token_type_ids = [ids + [tokenizer.pad_token_id] * (n - len(ids)) for ids in token_type_ids]
        # build sentence pairs - ignore samples that would overflow the sequence length
        sentence_pairs = [input_ids + input_ids[s:e] + [tokenizer.sep_token_id] for s, e in entity_spans if k + e - s + 1 <= n]
        sentence_pairs = [ids + [tokenizer.pad_token_id] * (n - len(ids)) for ids in sentence_pairs]

        # return items
        return list(zip(sentence_pairs, token_type_ids, labels))

    def preprocess(self, input_ids, token_type_ids, labels, tokenizer, device) -> dict:
        # move input ids and labels to device
        input_ids = input_ids.to(device) 
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        # build masks
        attention_mask = (input_ids != tokenizer.pad_token_id)
        # build keyword arguments for forward call
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }, labels
        
