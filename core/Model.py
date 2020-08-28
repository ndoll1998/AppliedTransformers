import torch
import transformers

class BaseModel(transformers.PreTrainedModel):
    """ Base Class for Models """

    # tokenizer type
    TOKENIZER_TYPE:type = transformers.BertTokenizer

    def build_feature_tensors(self, *item, seq_length:int, tokenizer:transformers.PreTrainedTokenizer) -> tuple:
        """ Build the feature tensors from a given data item. 
            It receives a data item and needs to return feature-tensors.
        """
        raise NotImplementedError()

    def preprocess(self, *batch, tokenizer) -> dict:
        """ Preprocess a batch from the dataset.
            Returns the keyword arguments for the forward call. 
        """
        raise NotImplementedError

    def preprocess_and_predict(self, *batch, tokenizer, device):
        """ Preprocess and predict a given batch
            Returns the output of the model as well as the target labels.
            Note that the target labels are not moved to the given cuda device.
        """
        # preprocess and move all tensors to device
        kwargs, labels = self.preprocess(*batch, tokenizer)
        kwargs = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}
        # predict
        return self.forward(**kwargs), labels