# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import base model
from .EntityClassificationModel import EntityClassificationModel
# import Bert Model and Tokenizer
from transformers import BertModel
from transformers import BertTokenizer
# import dataset
from ..datasets import EntityClassificationDataset
# import utils
from core.utils import align_shape, train_default_kwargs, eval_default_kwargs

class BertForEntityClassificationTokenizer(BertTokenizer):
    """ Tokenizer for the Bert Entity Classification Model """

    def __init__(self, *args, **kwargs) -> None:
        # initialize tokenizer
        BertTokenizer.__init__(self, *args, **kwargs)
        # add entity marker tokens
        self.add_tokens(['[e]', '[/e]'])

    @property
    def entity_token_id(self) -> int:
        return self.convert_tokens_to_ids('[e]')
    @property
    def _entity_token_id(self) -> int:
        return self.convert_tokens_to_ids('[/e]')


class BertForEntityClassification(BertModel, EntityClassificationModel):

    # set tokenizer type
    TOKENIZER_TYPE = BertForEntityClassificationTokenizer

    def __init__(self, config):
        # initialize bert model
        BertModel.__init__(self, config)
        # initialize classifier
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # initialize weights
        self.init_weights()

    def prepare(self, dataset:EntityClassificationDataset, tokenizer:BertForEntityClassificationTokenizer) -> None:
        # resize token embeddings to match the tokenizer
        self.resize_token_embeddings(len(tokenizer))

    @train_default_kwargs(max_entities=5)
    @eval_default_kwargs(seq_length=None, max_entities=None)
    def build_feature_tensors(self, input_ids, entity_spans, labels, seq_length, max_entities, tokenizer=None) -> tuple:

        if seq_length is not None:
            # remove entities that will be out of bounds after markers are added
            # entities are ordered by their occurances in the text (see EntityClassificationDataset)
            entity_spans = [(b, e) for i, (b, e) in enumerate(entity_spans, 1) if e + 2 * i < seq_length]
            labels = labels[:len(entity_spans)]

        # mark entities and build entity starts
        entity_starts = []
        for i in range(len(entity_spans) - 1, -1, -1):
            # mark current entity
            b, e = entity_spans[i]
            input_ids = input_ids[:b] + [tokenizer.entity_token_id] + input_ids[b:e] + [tokenizer._entity_token_id] + input_ids[e:]
            # get entity start id
            entity_starts.insert(0, b + 2 * i)

        # fill tensors
        input_ids = align_shape(input_ids, (seq_length,), tokenizer.pad_token_id) if seq_length is not None else input_ids
        entity_starts = align_shape(entity_starts, (max_entities,), -1) if max_entities is not None else entity_starts
        labels = align_shape(labels, (max_entities,), -1) if labels is not None else None
        # convert to tensors
        input_ids = torch.LongTensor([input_ids])
        entity_starts = torch.LongTensor([entity_starts])
        labels = torch.LongTensor([labels]) if labels is not None else None
        # return features tensors
        return input_ids, entity_starts, labels

    def preprocess(self, input_ids, entity_starts, labels, tokenizer) -> dict:
        # build masks
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        entity_mask = (labels != -1).long()
        # build keyword arguments for forward call
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_starts': entity_starts,
            'entity_mask': entity_mask,
            'labels': labels
        }, labels
        
    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        entity_starts=None,
        entity_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # pass through bert-model
        output = super(BertForEntityClassification, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )
        # build classifier input
        sequence_output = output[0]
        idx = torch.arange(sequence_output.size(0)).repeat(entity_starts.size(1), 1).t()
        feats = sequence_output[idx, entity_starts, :]
        # pass through classifier
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        # build outputs
        outputs = (logits,) + output[2:]

        # compute loss
        if labels is not None:
            # get valid labels and logits
            mask = labels >= 0
            logits, labels = logits[mask], labels[mask]
            # compute loss and add it to outputs
            loss = F.cross_entropy(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs


""" KnowBert Model """

# import KnowBert Model and utils
from external.KnowBert.src.kb.model import KnowBertModel
from external.utils import knowbert_build_caches_from_input_ids
# import knowledge bases to register them
import external.KnowBert.src.knowledge

class KnowBertForEntityClassification(BertForEntityClassification, KnowBertModel):

    def __init__(self, config) -> None:
        # initialize knowbert model
        KnowBertModel.__init__(self, config)
        # initialize classifier
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # initialize weights
        self.init_weights()

    def build_feature_tensors(self, *args, tokenizer=None, **kwargs) -> tuple:
        # build feature-tensors
        input_ids, entity_starts, labels = BertForEntityClassification.build_feature_tensors(self, *args, tokenizer=tokenizer, **kwargs)
        # build caches
        caches = knowbert_build_caches_from_input_ids(self, input_ids, tokenizer)
        # return feature tensors and caches
        return (input_ids, entity_starts, labels) + caches

    def preprocess(self, input_ids, entity_starts, labels, *caches, tokenizer) -> dict:
        # set caches
        self.set_valid_kb_caches(*caches)
        # preprocess
        return BertForEntityClassification.preprocess(self, input_ids, entity_starts, labels, tokenizer=tokenizer)