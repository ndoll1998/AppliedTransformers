import torch
# import base model
from .AspectOpinionExtractionModel import AspectOpinionExtractionModel
# import Bert Config and Model
from transformers import BertConfig
from transformers import BertForTokenClassification
# import utils
from core.utils import align_shape

class BertForAspectOpinionExtraction(AspectOpinionExtractionModel, BertForTokenClassification):

    def __init__(self, config:BertConfig):
        # number of labels to predict is 3+3 = 6
        # BIO-Scheme for aspects and opinions separately
        config.num_labels = 6
        # initialize model
        BertForTokenClassification.__init__(self, config)

    def prepare(self, input_ids, aspect_bio, opinion_bio, seq_length=None, tokenizer=None):

        if seq_length is not None:
            input_ids = align_shape(input_ids, (seq_length,), tokenizer.pad_token_id)
            aspect_bio = align_shape(aspect_bio, (seq_length,), -1) if aspect_bio is not None else None
            opinion_bio = align_shape(opinion_bio, (seq_length,), -1) if opinion_bio is not None else None

        # convert to tensors
        input_ids = torch.LongTensor([input_ids])
        aspect_bio = torch.LongTensor([aspect_bio]) if aspect_bio is not None else None
        opinion_bio = torch.LongTensor([opinion_bio]) if opinion_bio is not None else None
        # return feature tensors
        return input_ids, aspect_bio, opinion_bio

    def preprocess(self, input_ids, labels_a, labels_o, tokenizer) -> dict:
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
