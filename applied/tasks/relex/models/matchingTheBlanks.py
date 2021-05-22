import torch
import torch.nn as nn
import torch.nn.functional as F
# import base model and dataset item
from .base import RelExModel
from ..datasets.base import RelExDatasetItem
from applied.core.model import Encoder, InputFeatures
# import utils
from typing import Tuple
from dataclasses import dataclass
from applied.common.token import build_token_spans

@dataclass
class _MTB_InputFeatures(InputFeatures):
    e1e2_start:torch.LongTensor =None

class MatchingTheBlanks(RelExModel):

    def __init__(self, encoder:Encoder, num_labels:int, dropout:float =0.01) -> None:
        # initialize base model
        RelExModel.__init__(self, encoder=encoder)
        # add special tokens to tokenizer
        self.encoder.tokenizer.add_tokens(['[e1]', '[/e1]', '[e2]', '[/e2]', '[blank]'])
        self.encoder.resize_token_embeddings(len(encoder.tokenizer))
        # create classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * encoder.hidden_size, num_labels)
    
    def build_features_from_item(self, item:RelExDatasetItem) -> Tuple[_MTB_InputFeatures]:
        # tokenize text and build token spans
        tokens = self.encoder.tokenizer.tokenize(item.sentence)
        token_spans = build_token_spans(tokens, item.sentence)
        # find entity tokens
        S_span, T_span = item.source_entity_span, item.target_entity_span
        S_entity_token_idx = [i for i, (b, e) in enumerate(token_spans) if (S_span[0] <= b) and (e <= S_span[1])]
        T_entity_token_idx = [i for i, (b, e) in enumerate(token_spans) if (T_span[0] <= b) and (e <= T_span[1])]
        # no overlap between entities
        if len(set(S_entity_token_idx) & set(T_entity_token_idx)) > 0:
            return None
        # entity not found
        if (len(S_entity_token_idx) == 0) or (len(T_entity_token_idx) == 0):
            return None
        # find entity token spans
        S_entity_token_span = (S_entity_token_idx[0], S_entity_token_idx[-1] + 1)
        T_entity_token_span = (T_entity_token_idx[0], T_entity_token_idx[-1] + 1)

        # associate entities with markers
        entity_A = (["[e1]"], ["[/e1]"], *S_entity_token_span)
        entity_B = (["[e2]"], ["[/e2]"], *T_entity_token_span)
        # sort entities
        e1, e2 = sorted([entity_A, entity_B], key=lambda e: e[2])
        assert e1[3] <= e2[2] # make sure there is no overlap between entities
        # mark entities in tokens
        marked_tokens = tokens[:e1[2]] + e1[0] + tokens[e1[2]:e1[3]] + e1[1] + \
            tokens[e1[3]:e2[2]] + e2[0] + tokens[e2[2]:e2[3]] + e2[1] + \
            tokens[e2[3]:]
        # get entity starts
        # account for [CLS] at beginning of the sentence
        e1e2_start = (
            marked_tokens.index("[e1]") + 1, 
            marked_tokens.index("[e2]") + 1
        )
        # build input features
        return (_MTB_InputFeatures(
            text=item.sentence,
            tokens=["[CLS]"] + marked_tokens + ["[SEP]"],
            labels=item.relation_type,
            e1e2_start=e1e2_start
        ),)

    def truncate_feature(self, f:_MTB_InputFeatures, max_seq_length:int) -> _MTB_InputFeatures:
        # find entity markers in tokens
        e1_start, e2_start = f.e1e2_start
        e1_end, e2_end = f.tokens.index('[/e1]'), f.tokens.index('[/e2]')
        # sort by index
        (e1_start, e1_end), (e2_start, e2_end) = sorted(
            [(e1_start, e1_end), (e2_start, e2_end)], key=lambda x: x[0])
        # cut at both ends to keep entities in range of sequence length
        # check if entities are to far apart for sequence length
        if e2_end - e1_start > max_seq_length:
            return None
        # update feature
        off = max(0, e2_end - max_seq_length)
        f.tokens = f.tokens[off:off + max_seq_length]
        f.e1e2_start = (f.e1e2_start[0] - off, f.e1e2_start[1] - off)
        # return updated feature
        return f
        
    def build_target_tensors(self, features:Tuple[_MTB_InputFeatures]) -> Tuple[torch.Tensor]:
        return (torch.LongTensor([f.label_ids for f in features]),)

    def forward(self,
        # encoder inputs
        input_ids, 
        attention_mask=None, 
        token_type_ids=None,
        # additional inputs
        e1e2_start=None,
    ):
        # pass through base model
        sequence_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        # build classifier input
        idx = torch.arange(sequence_output.size(0)).repeat(2, 1).t()
        v1v2 = sequence_output[idx, e1e2_start, :].view(sequence_output.size(0), -1)
        # pass through classifier
        v1v2 = self.dropout(v1v2)
        logits = self.classifier(v1v2)
        # return logits
        return logits

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
