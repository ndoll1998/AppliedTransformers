# import base models
from transformers import BertForSequenceClassification
from .AspectBasedSentimentAnalysisModel import AspectBasedSentimentAnalysisModel

class BertForSentencePairClassification(AspectBasedSentimentAnalysisModel, BertForSequenceClassification):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def prepare(self, input_ids, aspects_token_ids, labels, tokenizer) -> list:
        n = len(input_ids)
        # remove padding tokens, entity spans and labels
        input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
        aspects_token_ids = [[t for t in tokens if t != tokenizer.pad_token_id] for tokens in aspects_token_ids]
        aspects_token_ids = [ids for ids in aspects_token_ids if len(ids) > 0]
        labels = [l for l in labels if l != -1]
        # check lengths
        assert len(aspects_token_ids) == len(labels)

        # add special tokens
        if input_ids[0] != tokenizer.cls_token_id:
            input_ids.insert(0, tokenizer.cls_token_id)
        if input_ids[-1] != tokenizer.sep_token_id:
            input_ids.append(tokenizer.sep_token_id)
        # remove special tokens from aspects tokens
        aspects_token_ids = [ids[1:] if ids[0] == tokenizer.cls_token_id else ids for ids in aspects_token_ids]
        aspects_token_ids = [ids[:-1] if ids[-1] == tokenizer.sep_token_id else ids for ids in aspects_token_ids]

        k = len(input_ids)
        # build token-type-ids - ignore samples that would overflow the sequence length
        token_type_ids = [[0] * k + [1] * (len(ids) + 1) for ids in aspects_token_ids if k + len(ids) + 1 <= n]
        token_type_ids = [ids + [tokenizer.pad_token_id] * (n - len(ids)) for ids in token_type_ids]
        # build sentence pairs - ignore samples that would overflow the sequence length
        sentence_pairs = [input_ids + ids + [tokenizer.sep_token_id] for ids in aspects_token_ids if k + len(ids) + 1 <= n]
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
            'token_type_ids': token_type_ids
        }, labels
        
