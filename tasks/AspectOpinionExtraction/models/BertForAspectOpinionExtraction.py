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
            'attention_mask': mask
        }, (labels_a, labels_o)

    def forward(self, 
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
        # return new outputs
        return outputs
