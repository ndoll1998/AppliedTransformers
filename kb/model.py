import re
# import pytorch
import torch
import torch.nn as nn
# import transformers
from transformers import BertModel, BertConfig
from transformers.modeling_bert import (
    BertForPreTraining, BertPreTrainingHeads, 
    BertEmbeddings, BertEncoder, BertPooler
)
# import kar and knowledge base
from .kar import KAR
from .knowledge import KnowledgeBase


""" KnowBert Encoder """

class KnowBertEncoder(BertEncoder):

    def __init__(self, config):
        # initialize Module
        super(KnowBertEncoder, self).__init__(config)
        # list of kb per layer
        self.kbs = nn.ModuleList([None for _ in range(self.config.num_hidden_layers)])


    # *** general ***

    def add_knowledge(self, layer:int, kb:KnowledgeBase, max_mentions=10, max_mention_span=5, max_candidates=10, threshold=None):
        """ add a knowledge bases in between layer and layer+1 """
        
        # check if kb is of correct type
        if not isinstance(kb, KnowledgeBase):
            raise RuntimeError("%s must inherit KnowledgeBase" % kb.__class__.__name__)
        # check if layer already has a kb
        if self.kbs[layer] is not None:
            raise RuntimeError("There already is a knowledge base at layer %i" % layer)
        
        # span-encoder-config
        span_encoder_config = BertConfig.from_dict({
            # "hidden_size": 300,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 1024
        })
        # span-attention-config
        span_attention_config = BertConfig.from_dict({
            # "hidden_size": 300,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 1024
        })

        # add knowledge base to layer
        self.kbs[layer] = KAR(
            kb, self.config, span_encoder_config, span_attention_config, 
            max_mentions=max_mentions, max_mention_span=max_mention_span,
            max_candidates=max_candidates, threshold=threshold
        )

        # return knowledge base
        return self.kbs[layer]

    def freeze_layers(self, layer:int):
        """ Freeze all parameters up to and including layer.
            This includes freezeing all knowledge bases up to but excluding the given layer
            
            Since the backward-graph will not reach any parameters before the encoder, 
            those parameters will also have no gradients
        """
        
        for n, p in self.named_parameters():
            # find layer index of parameter
            m = re.search(r".\d+.", n)
            l = int(m.group().replace('.', ''))
            # freeze if layer is before the given one
            if l <= layer:
                # exclude knowledge bases
                if n.startswith('kbs') and (l == layer):
                    # unfreeze
                    p.requires_grad_(True)
                else:
                    # freeze parameter
                    p.requires_grad_(False)
            else:
                # unfreeze
                p.requires_grad_(True)


    # *** caches ***

    def reset_kb_caches(self):
        """ reset all caches of all knowledge bases """
        for kb in self.kbs:
            if kb is not None:
                kb.reset_cache()

    def prepare_kbs(self, batch_tokens:list):
        """ prepare all knowledge bases for next forward pass.
            Basically computes and sets all caches for the given batch.

            Returns a list of mention-candidates dicts for each layer and each token-sequence

            return = ([dict, ..., dict], ..., [dict, ..., dict])
        """
        # reset and set caches
        self.reset_kb_caches()
        caches, candidates = zip(*[self.get_kb_caches(tokens) for tokens in batch_tokens])
        self.stack_kb_caches(*caches)
        # return all candidates
        return candidates

    def get_kb_caches(self, tokens:list):
        """ Get cache for each knowledge base from tokens.
            Return a list of caches, one for each knowledge base and None for layers without one.
            Also returns a mention-candidates dict for each layer.

            return = [cache, ..., cache], [dict, ..., dict]
        """        
        caches, candidates = zip(*[kb.get_cache(tokens) if kb is not None else (None, None) for kb in self.kbs])
        return caches, candidates

    def stack_kb_caches(self, *all_caches):
        """ Stack multiple caches for each knowledge base. 
            all_caches must be a list of caches where each list contains 
            one cache per knowledge base and None for layers without one

            all_caches = *([cache, ..., cache], ..., [cache, ..., cache])
        """
        # loop over all caches per knowledge base
        for kb, caches in zip(self.kbs, zip(*all_caches)):
            if kb is not None:
                assert None not in caches
                kb.stack_caches(*caches)


    # *** forward ***

    def forward(self, 
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_linking_scores=True,
        return_dict=False
    ):
        # prepare outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_linking_scores = () if output_linking_scores else None

        # pass through each layer
        for i, layer_module in enumerate(self.layer):
            # save all hidden states if asked for
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )

            else:
                # pass through layer
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            
            # get hidden state
            hidden_states = layer_outputs[0]

            # save attention values if asked for
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # apply knowledge base for layer if there is one
            if self.kbs[i] is not None:
                hidden_states, linking_scores = self.kbs[i].forward(hidden_states)
                # add linking scores to tuple
                if output_linking_scores:
                    all_linking_scores = all_linking_scores + (linking_scores,)

            # add None to linking scores tuple
            elif output_linking_scores:
                all_linking_scores = all_linking_scores + (None,)


        # add very last hidden state to tuple
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # return tuple
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, all_linking_scores] if v is not None)
        # return dict/output-class
        return BaseModelOutput(
            last_hidden_state=hidden_states, 
            hidden_states=all_hidden_states, 
            attentions=all_attentions
        )


""" KnowBert Models """

class KnowBertHelper(object):
    """ Define some basic methods that all KnowBert Models should have """

    def __init__(self, know_bert_encoder_instance):
        # save encoder instance without pytorch tracking it
        object.__setattr__(self, '_knowbert_encoder', know_bert_encoder_instance)

    @property
    def kbs(self):
        return self._knowbert_encoder.kbs
    
    def freeze_layers(self, layer:int):
        """ Freeze all layers after given index """
        self._knowbert_encoder.freeze_layers(layer)

    def add_kb(self, layer:int, kb:KnowledgeBase, *args, **kwargs):
        """ add a knowledge base layer and layer+1 """
        return self._knowbert_encoder.add_knowledge(layer, kb, *args, **kwargs)

    def reset_kb_caches(self):
        """ reset all caches """
        return self._knowbert_encoder.reset_kb_caches()

    def prepare_kbs(self, tokens:list):
        """ prepare all knowledge bases for next forward pass """
        return self._knowbert_encoder.prepare_kbs(tokens)

    def get_kb_caches(self, tokens):
        """ Get cache for each knowledge base from tokens.
            Return a list of caches, one for each knowledge base and None for layers without one 

            return = [cache, ..., cache]
        """  
        return self._knowbert_encoder.get_kb_caches(tokens)

    def stack_kb_caches(self, *caches):
        """ Stack multiple caches for each knowledge base. 
            all_caches must be a list of caches where each list contains 
            one cache per knowledge base and None for layers without one

            all_caches = *([cache, ..., cache], ..., [cache, ..., cache])
        """
        return self._knowbert_encoder.stack_kb_caches(*caches)


class KnowBert(BertModel, KnowBertHelper):
    """ Basic KnowBert Model as discribed in: "Knowledge Enhanced Contextual Word Representations"
        arxiv: https://arxiv.org/pdf/1909.04164.pdf
    """

    def __init__(self, config:dict):
        # dont call constructor of bert-model but instead
        # call the constructor of bert-model super class
        super(BertModel, self).__init__(config)

        # basically the constructor of bert-model but
        # using know-bert-encoder instead of bert-encoder
        self.embeddings = BertEmbeddings(config)
        self.encoder = KnowBertEncoder(config)
        self.pooler = BertPooler(config)
        # initialize weights
        self.init_weights()

        # initialize helper
        KnowBertHelper.__init__(self, self.encoder)

class KnowBertForPretraining(BertForPreTraining, KnowBertHelper):
    """ KnowBert for pretraining. 
        Basically BertForPreTraining but using KnowBert as model instead of standard BERT.
    """

    def __init__(self, config):
        # dont call constructor of BertPreTrainingModel 
        # but call it's super constructor
        super(BertForPreTraining, self).__init__(config)
        # create model and heads
        self.bert = KnowBert(config)
        self.cls = BertPreTrainingHeads(config)
        # initialize weights
        self.init_weights()

        # initialize helper
        KnowBertHelper.__init__(self, self.bert.encoder)
