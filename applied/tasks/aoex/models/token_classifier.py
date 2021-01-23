# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import from applied transformers
from .base import AOEx_Model
from ..datasets.base import AOEx_DatasetItem
from applied.core.model import Encoder, InputFeatures
# import utils
from applied.common import build_token_spans, build_bio_scheme, align_shape
from typing import Tuple

class TokenClassifier(AOEx_Model):

    def __init__(self, encoder:Encoder, dropout:float=0.01):
        AOEx_Model.__init__(self, encoder=encoder)
        # token classifier
        # number of labels to predict is 3+3 = 6
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hidden_size, 6)

    def build_features_from_item(self, item:AOEx_DatasetItem) -> Tuple[InputFeatures]:
        # tokenize everything
        tokens = self.encoder.tokenizer.tokenize(item.sentence)
        token_spans = build_token_spans(tokens, item.sentence)
        # create bio-schemes for aspects and opinions
        aspect_bio = build_bio_scheme(token_spans, item.aspect_spans)
        opinion_bio = build_bio_scheme(token_spans, item.opinion_spans)
        # return feature pair
        return (InputFeatures(
            text=item.sentence,
            tokens=["[CLS]"] + tokens + ["[SEP]"],
            labels=(
                [0] + aspect_bio + [0], 
                [0] + opinion_bio + [0]
            )
        ),)

    def truncate_feature(self, f:InputFeatures, max_seq_length:int) -> InputFeatures:
        # truncate both aspect and opinion bios
        f.tokens = f.tokens[:max_seq_length]
        f.labels = (f.labels[0][:max_seq_length], f.labels[1][:max_seq_length])
        return f

    def build_target_tensors(self, features:Tuple[InputFeatures])-> Tuple[torch.LongTensor]:
        # separate aspect and opinion bios
        aspect_bio, opinion_bio = zip(*(f.labels for f in features))
        # compute shape
        seq_length = max(map(len, aspect_bio + opinion_bio))
        shape = (len(features), seq_length)
        # align shapes to sequence length
        aspect_bio = align_shape(aspect_bio, shape, fill_value=-1)
        opinion_bio = align_shape(opinion_bio, shape, fill_value=-1)
        # return tensors
        return (torch.LongTensor(aspect_bio), torch.LongTensor(opinion_bio))

    def forward(self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):
        # pass through base model
        hidden_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        # pass through classifier
        output = self.dropout(hidden_output)
        logits = self.classifier(output)
        # separate aspect and opinion logits
        return logits[:, :, :3], logits[:, :, 3:]

    def loss(self, aspect_logits, opinion_logits, aspect_bio, opinion_bio):
        # flatten logits and targets
        flat_aspect_logits, flat_aspect_bio = aspect_logits.flatten(0, 1), aspect_bio.flatten()
        flat_opinion_logits, flat_opinion_bio = opinion_logits.flatten(0, 1), opinion_bio.flatten()
        # build masks
        aspect_mask = (flat_aspect_bio >= 0)
        opinion_mask = (flat_opinion_bio >= 0)
        # compute combined aspect and opinion loss
        return F.cross_entropy(flat_aspect_logits[aspect_mask], flat_aspect_bio[aspect_mask]) + \
                F.cross_entropy(flat_opinion_logits[opinion_mask], flat_opinion_bio[opinion_mask])
