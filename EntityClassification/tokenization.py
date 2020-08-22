# import base tokenizer
from transformers import BertTokenizer

class BertForEntityLinkingTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        # initialize tokenizer
        BertTokenizer.__init__(self, *args, **kwargs)
        # add entity marker tokens
        self.add_tokens(['[e]', '[/e]'])

    @property
    def entity_token_id(self):
        return self.convert_tokens_to_ids('[e]')
    @property
    def _entity_token_id(self):
        return self.convert_tokens_to_ids('[/e]')
