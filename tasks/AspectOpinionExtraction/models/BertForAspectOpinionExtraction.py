import torch
import torch.nn.functional as F
# import base model
from .AspectOpinionExtractionModel import AspectOpinionExtractionModel
# import Bert Config and Model
from transformers import BertConfig
from transformers import BertForTokenClassification

class BertForAspectOpinionExtraction(AspectOpinionExtractionModel, BertForTokenClassification):

    def __init__(self, config:BertConfig):
        # number of labels to predict is 3+3 = 6
        # BIO-Scheme for aspects and opinions separately
        config.num_labels = 6
        # initialize model
        BertForTokenClassification.__init__(self, config)

    def preprocess(self, input_ids, labels_a, labels_o, tokenizer, device) -> dict:
        # move all to device
        input_ids = input_ids.to(device)
        labels_a, labels_o = labels_a.to(device), labels_o.to(device)
        # create attention mask
        mask = (input_ids != tokenizer.pad_token_id)
        # return kwargs for forward call
        return {
            'input_ids': input_ids,
            'attention_mask': mask,
            'aspect_labels': labels_a,
            'opinion_labels': labels_o
        }, labels_a, labels_o

    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        aspect_labels=None,
        opinion_labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # predict
        outputs = BertForTokenClassification.forward(self,
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )
        # separate aspect and opinion predictions
        aspect_logits, opinion_logits = outputs[0][..., :3], outputs[0][..., 3:]
        outputs = (aspect_logits, opinion_logits) + outputs[1:]
        # check if any labels are passed
        if (aspect_labels is not None) or (opinion_labels is not None):
            # compute loss
            aspect_loss = F.cross_entropy(aspect_logits[attention_mask], aspect_labels[attention_mask]) if aspect_labels is not None else 0.0
            opinion_loss = F.cross_entropy(opinion_logits[attention_mask], opinion_labels[attention_mask]) if opinion_labels is not None else 0.0
            loss = aspect_loss + opinion_loss
            # build output tuple
            outputs = (loss,) + outputs
        # return new outputs
        return outputs
