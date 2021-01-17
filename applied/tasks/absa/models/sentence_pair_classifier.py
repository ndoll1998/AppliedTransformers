# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import from applied transformers
from .base import ABSA_Model
from ..datasets.base import ABSA_DatasetItem
from applied.core.model import Encoder, FeaturePair
# import utils
from typing import Tuple

class SentencePairClassifier(ABSA_Model):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def __init__(self, encoder:Encoder, num_labels:int, dropout:float=0.01):
        ABSA_Model.__init__(self, encoder=encoder)
        # classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hidden_size, num_labels)

    def build_features_from_item(self, item:ABSA_DatasetItem) -> Tuple[FeaturePair]:
        return tuple(FeaturePair(
                text=item.sentence + " [SEP] " + aspect,
                labels=label
            ) for aspect, label in zip(item.aspects, item.labels))

    def build_target_tensors(self, features:Tuple[FeaturePair], seq_length:int) -> Tuple[torch.LongTensor]:
        return (torch.LongTensor([f.labels for f in features]),)

    def forward(self, 
        input_ids, 
        attention_mask=None, 
        token_type_ids=None
    ):
        # pass through base model
        pooled_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1]
        # pass through classifier
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        # return
        return logits

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
