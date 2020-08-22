import torch
import torch.nn as nn
import torch.nn.functional as F
# import Bert Config and Model
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel


class BertForEntityClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForEntityClassification, self).__init__(config)
        # initialize bert and classifier
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # initialize weights
        self.init_weights()

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