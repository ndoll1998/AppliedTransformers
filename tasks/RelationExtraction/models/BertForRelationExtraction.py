import torch
import torch.nn as nn
import torch.nn.functional as F
# import base model and tokenizer
from .RelationExtractionModel import RelationExtractionModel
# import Bert Model and Tokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer

class BertForRelationExtractionTokenizer(BertTokenizer):
    """ Tokenizer for the Bert Entity Classification Model """

    def __init__(self, *args, **kwargs) -> None:
        # initialize tokenizer
        BertTokenizer.__init__(self, *args, **kwargs)
        # add entity marker tokens
        self.add_tokens(['[e1]', '[/e1]', '[e2]', '[/e2]', '[blank]'])

    @property
    def entity1_token_id(self):
        return self.convert_tokens_to_ids('[e1]')
    @property
    def entity2_token_id(self):
        return self.convert_tokens_to_ids('[e2]')
    @property
    def _entity1_token_id(self):
        return self.convert_tokens_to_ids('[/e1]')
    @property
    def _entity2_token_id(self):
        return self.convert_tokens_to_ids('[/e2]')
    @property
    def blank_token_id(self):
        return self.convert_tokens_to_ids('[blank]')


class BertForRelationExtraction(RelationExtractionModel, BertPreTrainedModel):
    """ Implementation of "Matching the Blanks: Distributional Similarity for Relation Learning"
        Paper: https://arxiv.org/abs/1906.03158
    """

    # set tokenizer type
    TOKENIZER_TYPE = BertForRelationExtractionTokenizer

    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        # initialize bert and classifier
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        # initialize weights
        self.init_weights()

    def prepare(self, input_ids:list, entity_span_A:tuple, entity_span_B:tuple, label:int, seq_length:int, tokenizer:BertForRelationExtractionTokenizer) -> tuple:
        # mark A as entity one and B as entity two
        entity_A = ([tokenizer.entity1_token_id], [tokenizer._entity1_token_id], *entity_span_A) 
        entity_B = ([tokenizer.entity2_token_id], [tokenizer._entity2_token_id], *entity_span_B) 
        # sort entities
        e1, e2 = sorted([entity_A, entity_B], key=lambda e: e[2])
        assert e1[3] <= e2[2]    # no overlap between entities

        # use sequence length
        if seq_length is not None:
            # check if entities are to far apart for sequence length
            if e2[3] - e1[2] >= seq_length - 4:
                return None
            # truncate input ids such that both entities are in bounds
            off = max(0, e2[3] - seq_length + 4)
            input_ids = input_ids[off:off + seq_length - 4]
            # update entity marker positions
            e1 = e1[:2] + (e1[2] - off, e1[3] - off)
            e2 = e2[:2] + (e2[2] - off, e2[3] - off)

        # mark entities in input-ids
        marked_input_ids = input_ids[:e1[2]] + e1[0] + input_ids[e1[2]:e1[3]] + e1[1] + \
            input_ids[e1[3]:e2[2]] + e2[0] + input_ids[e2[2]:e2[3]] + e2[1] + \
            input_ids[e2[3]:]

        # get entity start positions
        e1_e2_start = (marked_input_ids.index(tokenizer.entity1_token_id), marked_input_ids.index(tokenizer.entity2_token_id))

        if seq_length is not None:
            # fill input ids to reach sequence length
            assert len(marked_input_ids) <= seq_length
            marked_input_ids += [tokenizer.pad_token_id] * (seq_length - len(marked_input_ids))

        # convert to tensors
        marked_input_ids = torch.LongTensor([marked_input_ids])
        e1_e2_start = torch.LongTensor([e1_e2_start])
        label = torch.LongTensor([label]) if label is not None else None
        # return new item features
        return marked_input_ids, e1_e2_start, label

    def preprocess(self, input_ids, e1_e2_start, labels, tokenizer, device):
        # move to device
        input_ids = input_ids.to(device)
        e1_e2_start = e1_e2_start.to(device)
        labels = labels.to(device)
        # create attention mask
        mask = (input_ids != tokenizer.pad_token_id)
        # create keyword arguments
        return {
            'input_ids': input_ids,
            'attention_mask': mask,
            'e1_e2_start': e1_e2_start
        }, labels

    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        e1_e2_start=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # pass through bert
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]        
        # build classifier input
        idx = torch.arange(sequence_output.size(0)).repeat(2, 1).t()
        v1v2 = sequence_output[idx, e1_e2_start, :].view(sequence_output.size(0), -1)
        # pass through classifier
        v1v2 = self.dropout(v1v2)
        logits = self.classifier(v1v2)
        # build outputs
        outputs = (logits,) + outputs[2:]

        # return output
        return outputs
