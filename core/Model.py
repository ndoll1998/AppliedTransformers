# import torch and transformers
import torch
import transformers

class BaseModel(transformers.PreTrainedModel):
    """ Base Class for Models """

    # tokenizer type
    TOKENIZER_TYPE:type = transformers.BertTokenizer

    def prepare(self, *item, tokenizer:transformers.PreTrainedTokenizer):
        """ Prepare a dataset item for the model. 
            This function is called during the dataset creation. 
            It receives a dataset item and needs to return feature-tensors.
        """
        return (torch.tensor([f]) for f in item)

    def preprocess(self, *batch, tokenizer, device:str) -> dict:
        """ Preprocess a batch from the dataset.
            Returns the keyword arguments for the forward call. 
        """
        raise NotImplementedError