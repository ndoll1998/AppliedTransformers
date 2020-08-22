import torch
import torch.nn as nn
import torch.nn.functional as F
# import Bert Config and Model
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel

class BertForRelationExtraction(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForRelationExtraction, self).__init__(config)
        # initialize bert and classifier
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        # initialize weights
        self.init_weights()

    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        e1_e2_start=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
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

        # compute loss
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs = (loss,) + outputs

        # return output
        return outputs