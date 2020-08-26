# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import base model
from .EntityClassificationModel import EntityClassificationModel
# import Bert Model and Tokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer


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


class BertForEntityClassification(EntityClassificationModel, BertPreTrainedModel):

    # set tokenizer type
    TOKENIZER_TYPE = BertForEntityClassificationTokenizer

    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        # initialize bert and classifier
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # initialize weights
        self.init_weights()

    def prepare(self, input_ids, entity_spans, labels, tokenizer) -> tuple:

        n = len(entity_spans)
        # ignore unvalid entity spans
        entity_spans = [(b, e) for b, e in entity_spans if b != -1]
        labels = [l for l in labels if l != -1]
        assert len(entity_spans) == len(labels)

        # sort entities
        sort_idx = sorted(range(len(entity_spans)), key=lambda i: entity_spans[i][0])
        entity_spans = [entity_spans[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]
        # remove entities that will be out of bounds after markers are added
        entity_spans = [(b, e) for i, (b, e) in enumerate(entity_spans, 1) if e + 2 * i < len(input_ids)]
        labels = labels[:len(entity_spans)]

        # mark entities in reverse order and build entity starts
        entity_starts = []
        for i in range(len(entity_spans) - 1, -1, -1):
            # mark current entity
            b, e = entity_spans[i]
            input_ids = input_ids[:b] + [tokenizer.entity_token_id] + input_ids[b:e] + [tokenizer._entity_token_id] + input_ids[e:]
            # get entity start id
            entity_starts.insert(0, b + 2 * i)

        # truncate to keep sequence length
        input_ids = input_ids[:-2 * len(entity_starts)] if len(entity_starts) > 0 else input_ids
        # pad to fill tensors
        entity_starts += [-1] * (n - len(entity_starts))
        labels += [-1] * (n - len(labels))
        # return new features
        return [(input_ids, entity_starts, labels)]

    def preprocess(self, input_ids, entity_starts, labels, tokenizer, device) -> dict:
        # move input ids and labels to device
        input_ids = input_ids.to(device) 
        labels = labels.to(device)
        entity_starts = entity_starts.to(device)
        # build masks
        attention_mask = (input_ids != tokenizer.pad_token_id)
        entity_mask = (labels != -1)
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
        output = self.bert(
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
            loss = F.cross_entropy(logits[entity_mask], labels[entity_mask])
            outputs = (loss,) + outputs

        return outputs
