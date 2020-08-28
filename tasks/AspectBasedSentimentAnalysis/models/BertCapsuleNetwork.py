import torch
import torch.nn as nn
import torch.nn.functional as F
# import base models
from transformers import BertModel, BertConfig
from .AspectBasedSentimentAnalysisModel import AspectBasedSentimentAnalysisModel
# import utils
from core.utils import align_shape


""" Custom Configuration """

class BertCapsuleNetworkConfig(BertConfig):

    def __init__(self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        capsule_size=300,
        loss_smooth=0.1,
        loss_lambda=0.6,
        **kwargs
    ):
        # initialize config
        BertConfig.__init__(self,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        # save capsule size
        self.capsule_size = capsule_size
        # save loss parameters
        self.loss_smooth = loss_smooth
        self.loss_lambda = loss_lambda


""" Bilinear Attention Module """

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

""" Bert Capsule Module """

def squash(x, dim=-1):
    squared = (x * x).sum(dim=dim, keepdim=True)
    scale = torch.sqrt(squared) / (1.0 + squared)
    return scale * x

class BertCapsuleNetwork(AspectBasedSentimentAnalysisModel, BertModel):
    """ "A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis"
        Paper: https://www.aclweb.org/anthology/D19-1654/
    """

    # set config class for bert model
    config_class = BertCapsuleNetworkConfig

    def __init__(self, config:BertConfig):
        # initialize bert model
        BertModel.__init__(self, config)
        # aspect transform
        self.aspect_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.capsule_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        # sentence transform
        self.sentence_transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.capsule_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        # attention
        self.norm_attention = BilinearAttention(config.capsule_size, config.capsule_size, config.attention_probs_dropout_prob)
        # capsule
        self.guide_capsule = nn.Parameter(torch.Tensor(config.num_labels, config.capsule_size))
        self.guide_weight = nn.Parameter(torch.Tensor(config.capsule_size, config.capsule_size))
        # projection
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.capsule_projection = nn.Linear(config.hidden_size, config.hidden_size * config.num_labels)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # reset parameters
        self._reset_parameters()

    def _reset_parameters(self):
        # randomize parameters
        nn.init.xavier_uniform_(self.guide_capsule)
        nn.init.xavier_uniform_(self.guide_weight)

    def build_feature_tensors(self, input_ids, aspects_token_ids, labels, seq_length=None, tokenizer=None) -> list:
        # one label per entity span
        assert (labels is None) or (len(aspects_token_ids) == len(labels))

        # add special tokens
        if input_ids[0] != tokenizer.cls_token_id:
            input_ids.insert(0, tokenizer.cls_token_id)
        if input_ids[-1] != tokenizer.sep_token_id:
            input_ids.append(tokenizer.sep_token_id)
        # remove special tokens from aspects tokens
        aspects_token_ids = [ids[1:] if ids[0] == tokenizer.cls_token_id else ids for ids in aspects_token_ids]
        aspects_token_ids = [ids[:-1] if ids[-1] == tokenizer.sep_token_id else ids for ids in aspects_token_ids]

        k = len(input_ids)
        # build overflow mask
        mask = [(seq_length is None) or (k + len(ids) + 1 <= seq_length) for ids in aspects_token_ids]
        # build token-type-ids and sentence pairs - ignore samples that would overflow the sequence length
        token_type_ids = [[0] * k + [1] * (len(ids) + 1) for ids, valid in zip(aspects_token_ids, mask) if valid]
        sentence_pairs = [input_ids + ids + [tokenizer.sep_token_id] for ids, valid in zip(aspects_token_ids, mask) if valid]
        # remove labels for examples that are out of bounds
        if labels is not None:
            labels = [l for l, valid in zip(labels, mask) if valid]

        # choose minimal sequence length to fit all examples
        if seq_length is None:
            seq_length = max((len(ids) for ids in sentence_pairs), default=0)
        # convert to tensors
        sentence_pairs = torch.LongTensor(align_shape(sentence_pairs, (len(sentence_pairs), seq_length), tokenizer.pad_token_id))
        token_type_ids = torch.LongTensor(align_shape(token_type_ids, (len(token_type_ids), seq_length), tokenizer.pad_token_id))
        labels = torch.LongTensor(labels) if labels is not None else None
        # return items
        return sentence_pairs, token_type_ids, labels

    def preprocess(self, input_ids, token_type_ids, labels, tokenizer) -> dict:
        # build masks
        attention_mask = (input_ids != tokenizer.pad_token_id)
        # build keyword arguments for forward call
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }, labels
        
    def forward(self, input_ids, attention_mask, token_type_ids, *args, labels=None, **kwargs):
        # encode 
        outputs = BertModel.forward(self, input_ids, attention_mask, token_type_ids, *args, **kwargs)
        sequence_output = outputs[0]
        # create sentence and aspect masks
        sentence_mask = attention_mask & (token_type_ids == 0)
        aspect_mask = attention_mask & (token_type_ids == 1)
        # get clean sentence and aspects
        sentence = sequence_output.masked_fill(~sentence_mask.unsqueeze(-1), 0)
        aspects = sequence_output.masked_fill(~aspect_mask.unsqueeze(-1), 0)
        # average pooling of aspects
        pooled_aspects = aspects.sum(dim=-2) / aspect_mask.sum(dim=-1, keepdim=True).float()
        # primary/sentence capsule layer
        encoded_sentence = self.sentence_transform(sentence)
        primary_capsule = squash(encoded_sentence, dim=-1)
        # secondary/aspects capsule layer
        encoded_aspects = self.aspect_transform(pooled_aspects)
        secondary_capsule = squash(encoded_aspects, dim=-1)
        # aspect-aware normalization
        norm_weight = self.norm_attention.get_attention_weight(secondary_capsule, primary_capsule, sentence_mask)
        # capsule guided routing
        category_capsule = self.capsule_gruided_routing(primary_capsule, norm_weight)
        category_capsule_norm = (category_capsule * category_capsule).sum(dim=-1)
        category_capsule_norm = torch.sqrt(category_capsule_norm)
        # add to outputs
        logits = category_capsule_norm
        outputs = (logits,) + outputs[1:]

        # compute maximum margin loss
        if labels is not None:
            # build one-hot matrix
            one_hot = torch.zeros_like(logits).to(logits.device)
            one_hot = one_hot.scatter(1, labels.unsqueeze(-1), 1)
            # compute loss
            a = torch.relu(1 - self.config.loss_smooth - logits)
            b = torch.relu(logits - self.config.loss_smooth)
            loss = one_hot * a * a + self.config.loss_lambda * (1 - one_hot) * b * b
            loss = loss.sum(dim=1).mean()
            # add to outputs
            outputs = (loss,) + outputs

        # return
        return outputs

    def capsule_gruided_routing(self, primary_capsule, norm_weight):
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
    
        