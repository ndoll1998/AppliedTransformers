# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import from applied transformers
from .base import EC_Model
from ..datasets.base import EC_DatasetItem
from applied.core.model import Encoder, FeaturePair
# model is based on ABSA's SentencePairClassifier
from applied.tasks.absa.models import SentencePairClassifier as ABSA_SentencePairClassifier
# import utils
from typing import Tuple

class SentencePairClassifier(ABSA_SentencePairClassifier, EC_Model):
    """ Implementation of "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
        Paper: https://arxiv.org/abs/1903.09588
    """

    def build_features_from_item(self, item:EC_DatasetItem) -> Tuple[FeaturePair]:
        return tuple(FeaturePair(
                text=item.sentence + " [SEP] " + item.sentence[b:e],
                labels=label
            ) for (b, e), label in zip(item.entity_spans, item.labels))
