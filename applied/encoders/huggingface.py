import torch
import transformers
from applied.core.model import Encoder
from applied.common.nested import align_shape

class HuggingfaceModel(transformers.PreTrainedModel, Encoder):

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def init_tokenizer_from_pretrained(self, *args, **kwargs):
        # initialize tokenizer
        assert self.__class__.TOKENIZER_TYPE is not None
        self.tokenizer = self.__class__.TOKENIZER_TYPE.from_pretrained(*args, **kwargs)

    def build_feature_tensors(self, features:tuple):
        # build all input ids
        input_ids = tuple(f.token_ids(self.tokenizer) for f in features)
        # choose minimal sequence length to fit all features
        seq_length = max(map(len, input_ids))
        shape = (len(features), seq_length)

        # build attention masks
        attention_mask = [[1] * len(ids) for ids in input_ids]
        # build token type ids
        sep = self.tokenizer.sep_token_id
        first_sep_idx = [(ids.index(sep) + 1) if sep in ids else len(ids) for ids in input_ids]
        token_type_ids = [[0] * idx + [1] * (len(ids) - idx) for idx, ids in zip(first_sep_idx, input_ids)]

        # match shape
        input_ids = align_shape(input_ids, shape, fill_value=self.tokenizer.pad_token_id)
        attention_mask = align_shape(attention_mask, shape, fill_value=0)
        token_type_ids = align_shape(token_type_ids, shape, fill_value=0)
        # build tensors
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        # return tensors
        return input_ids, attention_mask, token_type_ids

# create simple model types
BERT = type("BERT", (transformers.BertModel, HuggingfaceModel), {'TOKENIZER_TYPE': transformers.BertTokenizer})
