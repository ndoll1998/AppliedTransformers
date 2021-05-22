# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import applied transformers
from .base import ABSA_Model
from ..datasets.base import ABSA_Dataset, ABSA_DatasetItem
from applied.core.model import Encoder, InputFeatures
# import utils
from applied.common import align_shape
from typing import Tuple

""" Bi-Linear Attention """

class BilinearAttention(nn.Module):

    def __init__(self, query_size, key_size, dropout=0):
        super(BilinearAttention, self).__init__()
        # create weight and dropout layer
        self.weights = nn.Parameter(torch.FloatTensor(query_size, key_size))
        self.dropout = nn.Dropout(dropout)
        # randomize weights
        nn.init.xavier_uniform_(self.weights)

    def get_attention_weight(self, query, key, mask=None):
        # compute attention scores
        score = self.score(query, key)
        # apply mask and softmax
        if mask is not None:
            score = score.masked_fill(~mask, -10000)
        weight = F.softmax(score, dim=-1)
        # apply dropout
        return self.dropout(weight)

    def forward(self, query, key, value, mask=None):
        # compute attention weight
        weight = self.get_attention_weight(query, key, mask)
        # compute output
        return weight @ value, weight

    def score(self, query, key):
        # compute score
        return ((query @ self.weights).unsqueeze(-1) * key.transpose(1, 2)).sum(-2)

""" Capsule Network """

def squash(x, dim=-1):
    squared = (x * x).sum(dim=dim, keepdim=True)
    scale = torch.sqrt(squared) / (1.0 + squared)
    return scale * x

class CapsuleNetwork(ABSA_Model):
    """ "A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis"
        Paper: https://www.aclweb.org/anthology/D19-1654/
    """

    def __init__(self, 
        encoder:Encoder, 
        num_labels:int, 
        capsule_size:int =300,
        loss_smooth:float =0.1,
        loss_lambda:float =0.6,
        dropout_prob:float =0.1
    ) -> None:
        ABSA_Model.__init__(self, encoder=encoder)
        # loss hyperparameters
        self.loss_smooth = loss_smooth
        self.loss_lambda = loss_lambda
        # aspect transform
        self.aspect_transform = nn.Sequential(
            nn.Linear(encoder.hidden_size, capsule_size),
            nn.Dropout(dropout_prob)
        )
        # sentence transform
        self.sentence_transform = nn.Sequential(
            nn.Linear(encoder.hidden_size, capsule_size),
            nn.Dropout(dropout_prob)
        )
        # attention
        self.norm_attention = BilinearAttention(capsule_size, capsule_size, dropout_prob)
        # capsule
        self.guide_capsule = nn.Parameter(torch.Tensor(num_labels, capsule_size))
        self.guide_weight = nn.Parameter(torch.Tensor(capsule_size, capsule_size))
        # projection
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(encoder.hidden_size, encoder.hidden_size * num_labels)
        self.dropout = nn.Dropout(dropout_prob)

        # reset parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # randomize parameters
        nn.init.xavier_uniform_(self.guide_capsule)
        nn.init.xavier_uniform_(self.guide_weight)

    @torch.no_grad()
    def _init_guide_capsule(self, labels):
        self.eval()
        # tokenize labels
        label_tokens = [self.encoder.tokenizer.tokenize(label) for label in labels]
        label_ids = [self.encoder.tokenizer.convert_tokens_to_ids(tokens) for tokens in label_tokens]
        # create input ids for model
        shape = (len(labels), max((len(ids) for ids in label_ids)))
        input_ids = align_shape(label_ids, shape, self.encoder.tokenizer.pad_token_id)
        # create input tensors
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor((input_ids != self.encoder.tokenizer.pad_token_id).long())
        input_ids, attention_mask = input_ids.to(self.encoder.device), attention_mask.to(self.encoder.device)
        # pass through model
        label_embed = self.encoder.forward(input_ids, attention_mask=attention_mask)[0]
        label_embed = self.sentence_transform(label_embed)
        # compute average over timesteps
        label_embed = label_embed.sum(dim=1) / attention_mask.sum(dim=1, keepdims=True).float()
        # apply label embeddings
        self.guide_capsule.data.copy_(label_embed)

    def prepare(self, dataset:ABSA_Dataset) -> None:
        # initialize guide capsule
        self._init_guide_capsule(dataset.LABELS)

    def build_features_from_item(self, item:ABSA_DatasetItem) -> Tuple[InputFeatures]:
        return tuple(
            InputFeatures(
                text="[CLS]" + item.sentence + "[SEP]" + aspect + "[SEP]",
                labels=label
            ) for aspect, label in zip(item.aspects, item.labels)
        )

    def build_target_tensors(self, features:Tuple[InputFeatures]) -> Tuple[torch.Tensor]:
        return (torch.LongTensor([f.label_ids for f in features]),)

    def capsule_guided_routing(self, primary_capsule, norm_weight):
        # build guide matrix
        guide_capsule = squash(primary_capsule)
        guide_matrix = (primary_capsule @ self.guide_weight) @ self.guide_capsule.transpose(0, 1)
        guide_matrix = F.softmax(guide_matrix, dim=-1)
        guide_matrix = guide_matrix * norm_weight.unsqueeze(-1) * self.scale
        # build category capsule
        category_capsule = guide_matrix.transpose(1, 2) @ primary_capsule
        category_capsule = self.dropout(category_capsule)
        category_capsule = squash(category_capsule)
        # return
        return category_capsule

    def forward(self, input_ids, attention_mask, token_type_ids):
        # pass through encoder
        sequence_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        # create sentence and aspect masks
        sent_mask = attention_mask.bool() & (token_type_ids == 0)
        aspects_mask = attention_mask.bool() & (token_type_ids == 1).bool()
        # get clean sentence and aspects
        sent = sequence_output.masked_fill(~sent_mask.unsqueeze(-1), 0)
        aspects = sequence_output.masked_fill(~aspects_mask.unsqueeze(-1), 0)
        # average pool over aspect encodings
        pooled_aspects = aspects.sum(dim=-2) / aspects_mask.sum(dim=-1, keepdims=True).float()
        # primary/sentence capsule layer
        encoded_sent = self.sentence_transform(sent)
        primary_capsule = squash(encoded_sent, dim=-1)
        # secondary/aspects capsule layer
        encoded_aspects = self.aspect_transform(pooled_aspects)
        secondary_capsule = squash(encoded_aspects, dim=-1)
        # aspect-aware normalization
        norm_weight = self.norm_attention.get_attention_weight(secondary_capsule, primary_capsule, sent_mask)
        # capsule guided routing
        category_capsule = self.capsule_guided_routing(primary_capsule, norm_weight)
        category_capsule_norm = (category_capsule * category_capsule).sum(dim=-1)
        category_capsule_norm = torch.sqrt(category_capsule_norm)
        # return logits
        return category_capsule_norm

    def loss(self, logits, labels):
        # build one-hot matrix
        one_hot = torch.zeros_like(logits).to(logits.device)
        one_hot = one_hot.scatter(1, labels.unsqueeze(-1), 1)
        # compute loss
        a = torch.max(torch.zeros_like(logits), 1 - self.loss_smooth - logits)
        b = torch.max(torch.zeros_like(logits), logits - self.loss_smooth)
        loss = one_hot * a * a + self.loss_lambda * (1 - one_hot) * b * b
        loss = loss.sum(dim=1).mean()
        # add to outputs
        return loss
 
