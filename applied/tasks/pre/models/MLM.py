# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import applied transformers
from .base import PretrainModel
from ..datasets.base import PretrainDatasetItem
from applied.core.model import Encoder, InputFeatures
# import utils
from applied.common.nested import align_shape
from dataclasses import dataclass
from math import ceil
from random import shuffle, random, randint
from typing import Tuple
from tqdm import tqdm

class GeLU(nn.Module):
    """ Helper Module implementing a fast version of the gelu activation function """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def build_mlm_features(
    # text and tokenizer
    tokens:list,
    tokenizer:object,
    # parameters
    mask_prob:float, 
    max_pred_per_seq:int, 
    whole_word_masking:bool
) -> InputFeatures:
    # get tokenizer vocab
    vocab = list(tokenizer.vocab.keys())
    # get special tokens
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    MASK = tokenizer.mask_token
    # get all masking candidate indices
    candidates = []
    for i, token in enumerate(tokens):
        # ignore tokens
        if token in (CLS, SEP):
            continue
        # add index to candidate list
        if whole_word_masking and (len(candidates) > 0) and token.startswith('##'):
            candidates[-1].append(i)
        else:
            candidates.append([i])
    # shuffle
    shuffle(candidates)

    # get the number tokens to mask
    n = ceil(len(tokens) * mask_prob)
    n = min(max_pred_per_seq, max(1, n))
    # build masked tokens and targets
    targets = [-1] * len(tokens)
    n_masked = 0
    for idx in candidates:
        # enough tokens selected already
        if n_masked >= n:
            break
        # maximum number of masked tokens would be exceeded
        if n_masked + len(idx) > n:
            continue
        
        # mask tokens corresponding to idx
        for i in idx:
            # update mask
            targets[i] = tokenizer.convert_tokens_to_ids(tokens[i:i+1])[0]
            # choose replacement for token
            r = random()
            if r < 0.1:
                # keep token
                pass
            elif r < 0.2:
                # replace it with a random word
                tokens[i] = vocab[randint(0, len(vocab)-1)]
            else:
                tokens[i] = MASK
        # update the number of already masked tokens
        n_masked += len(idx)
    
    return InputFeatures(
        text=None,
        labels=None,
        tokens=tokens,
        label_ids=targets
    )

class MLM_Head(PretrainModel):
    """ Masked Language Modeling Head without Next Sentence Prediction """    

    def __init__(self, encoder:Encoder):
        PretrainModel.__init__(self, encoder=encoder)
        # mlm head
        self.decoder = nn.Sequential(
            nn.Linear(encoder.hidden_size, encoder.hidden_size),
            GeLU(),
            nn.LayerNorm(encoder.hidden_size, eps=1e-5),
            nn.Linear(encoder.hidden_size, len(encoder.tokenizer))
        )

    def build_features_from_item(self, item:PretrainDatasetItem) -> Tuple[InputFeatures]:
        # get tokenizer vocabulary
        vocab = list(self.encoder.tokenizer.vocab)
        # get special tokens
        CLS = self.encoder.tokenizer.cls_token
        SEP = self.encoder.tokenizer.sep_token
        MASK = self.encoder.tokenizer.mask_token
        # tokenize documents
        docs = item.documents
        n = sum(map(len, docs))
        sents = (sent for doc in docs for sent in doc)
        sents = (self.encoder.tokenizer.tokenize(sent) for sent in sents)
        sents = ([CLS] + tokens + [SEP] for tokens in sents)
        # create input features
        return tuple(
            build_mlm_features(
                tokens=tokens,
                tokenizer=self.encoder.tokenizer,
                mask_prob=0.15,
                max_pred_per_seq=20,
                whole_word_masking=True
            ) for tokens in tqdm(sents, total=n, ascii=True, desc="Building Features", leave=False)
        )

    def truncate_feature(self, f:InputFeatures, max_seq_length:int) -> InputFeatures:
        # truncate tokens and targets
        f.tokens = align_shape(f.tokens, shape=(max_seq_length,), fill_value=self.encoder.tokenizer.pad_token)
        f.label_ids = align_shape(f.label_ids, shape=(max_seq_length,), fill_value=-1)
        return f

    def build_target_tensors(self, features:Tuple[InputFeatures]) -> Tuple[torch.LongTensor]:
        return (torch.LongTensor([f.label_ids for f in features]),)
        
    def forward(self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):
        # apply encoder
        sequence_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        # apply decoder
        logits = self.decoder(sequence_output)
        return logits

    def loss(self, logits, targets):
        return F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=-1)
