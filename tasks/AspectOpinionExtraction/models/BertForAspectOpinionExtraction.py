import torch
import torch.nn.functional as F
# import base model
from .AspectOpinionExtractionModel import AspectOpinionExtractionModel
# import Bert Config and Model
from transformers import BertConfig
from transformers import BertForTokenClassification
# import utils
from core.utils import align_shape

class BertForAspectOpinionExtraction(BertForTokenClassification, AspectOpinionExtractionModel):

    def __init__(self, config:BertConfig) -> None:
        # number of labels to predict is 3+3 = 6
        # BIO-Scheme for aspects and opinions separately
        config.num_labels = 6
        # initialize model
        BertForTokenClassification.__init__(self, config)

    def build_feature_tensors(self, input_ids, aspect_bio, opinion_bio, seq_length=None, tokenizer=None):

        # fill sequence length
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
            'attention_mask': mask,
            'aspect_labels': labels_a,
            'opinion_labels': labels_o
        }, (labels_a, labels_o)

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
        outputs = super(BertForAspectOpinionExtraction, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )
        # separate aspect and opinion predictions
        aspect_logits, opinion_logits = outputs[0][..., :3], outputs[0][..., 3:]
        outputs = (aspect_logits, opinion_logits) + outputs[1:]

        loss = 0
        # compute aspect loss
        if aspect_labels is not None:
            mask = aspect_labels >= 0
            loss += F.cross_entropy(aspect_logits[mask], aspect_labels[mask])
        # compute opinion loss
        if opinion_labels is not None:
            mask = opinion_labels >= 0
            loss += F.cross_entropy(opinion_logits[mask], opinion_labels[mask])
        # add loss to output
        if (aspect_labels is not None) or (opinion_labels is not None):
            outputs = (loss,) + outputs

        # return new outputs
        return outputs


""" KnowBert Model """

# import KnowBert Model and utils
from external.KnowBert.src.kb.model import KnowBertForTokenClassification
from external.KnowBert.src.kb.configuration import KnowBertConfig
from external.utils import knowbert_build_caches_from_input_ids
# import knowledge bases to register them
import external.KnowBert.src.knowledge

class KnowBertForAspectOpinionExtraction(BertForAspectOpinionExtraction, KnowBertForTokenClassification):

    def __init__(self, config:KnowBertConfig) -> None:
        # number of labels to predict is 3+3 = 6
        # BIO-Scheme for aspects and opinions separately
        config.num_labels = 6
        # initialize model
        KnowBertForTokenClassification.__init__(self, config)

    def build_feature_tensors(self, *args, tokenizer, **kwargs):
        # build feature tensors
        input_ids, aspect_bio, opinion_bio = BertForAspectOpinionExtraction.build_feature_tensors(self, *args, tokenizer=tokenizer, **kwargs)
        # build caches
        caches = knowbert_build_caches_from_input_ids(self, input_ids, tokenizer)
        # return features and caches
        return (input_ids, aspect_bio, opinion_bio) + caches

    def preprocess(self, input_ids, labels_a, labels_o, *caches, tokenizer) -> dict:
        # set caches
        self.set_valid_kb_caches(*caches)
        # preprocess
        return BertForAspectOpinionExtraction.preprocess(self, input_ids, labels_a, labels_o, tokenizer=tokenizer)
