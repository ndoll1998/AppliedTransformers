# import transformers
import transformers

class BaseModel(transformers.PreTrainedModel):
    """ Base Class for Models """

    # tokenizer type
    TOKENIZER_TYPE:type = transformers.BertTokenizer

    def prepare(self, *item, seq_length:int, tokenizer:transformers.PreTrainedTokenizer) -> tuple:
        """ Prepare a data item for the model. 
            It receives a data item and needs to return feature-tensors.
        """
        raise NotImplementedError()

    def preprocess(self, *batch, tokenizer, device:str) -> dict:
        """ Preprocess a batch from the dataset.
            Returns the keyword arguments for the forward call. 
        """
        raise NotImplementedError