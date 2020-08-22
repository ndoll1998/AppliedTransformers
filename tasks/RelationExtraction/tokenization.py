# import base tokenizer
from transformers import BertTokenizer


class BertForRelationExtractionTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        # initialize tokenizer
        BertTokenizer.__init__(self, *args, **kwargs)
        # add entity marker tokens
        self.add_tokens(['[e1]', '[/e1]', '[e2]', '[/e2]', '[blank]'])


    @property
    def entity1_token_id(self):
        return self.convert_tokens_to_ids('[e1]')
    @property
    def entity2_token_id(self):
        return self.convert_tokens_to_ids('[e2]')
    @property
    def _entity1_token_id(self):
        return self.convert_tokens_to_ids('[/e1]')
    @property
    def _entity2_token_id(self):
        return self.convert_tokens_to_ids('[/e2]')
    @property
    def blank_token_id(self):
        return self.convert_tokens_to_ids('[blank]')
