# import torch
import torch
# import base model and dataset
from .Model import BaseModel
from .Dataset import BaseDataset


class BasePredictor(object):
    """ Base Class for Predictors """

    # base model and dataset types
    BASE_MODEL_TYPE = BaseModel
    BASE_DATASET_TYPE = BaseDataset

    def __init__(self,
        # model and tokenizer
        model_type:type =None,
        pretrained_name:str =None,
        model_kwargs:dict ={},
        device:str ='cpu',
        # dataset
        dataset_type:type =None
    ):
        # save values
        self.device = device
        self.pretrained_name = pretrained_name
        # create tokenizer
        self.tokenizer = model_type.TOKENIZER_TYPE.from_pretrained(pretrained_name)
        # check model type
        if not issubclass(model_type, self.__class__.BASE_MODEL_TYPE):
            raise ValueError("Model Type %s must inherit %s!" % (model_type.__name__, self.__class__.BASE_MODEL_TYPE.__name__))
        # create model
        self.model = model_type.from_pretrained(pretrained_name, **model_kwargs).to(device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        # check dataset type
        if not issubclass(dataset_type, self.__class__.BASE_DATASET_TYPE):
            raise ValueError("Dataset Type %s must inherit %s!" % (dataset_type.__name__, self.__class__.BASE_DATASET_TYPE.__name__))
        self.dataset_type = dataset_type

    def __call__(self, *args, **kwargs):
        # forward to prediction
        return self.predict(*args, **kwargs)

    @torch.no_grad()
    def predict(self, text, *args, **kwargs):
        # build and prepare item
        item = self.dataset_type.build_dataset_item(text, *args, **kwargs, tokenizer=self.tokenizer)
        item = self.model.build_feature_tensors(*item, tokenizer=self.tokenizer)
        # predict and postprocess
        outputs, _ = self.model.preprocess_and_predict(*item, tokenizer=self.tokenizer, device=self.device)
        return self.postprocess(*outputs)

    def postprocess(self, *outputs):
        """ Post-process model outputs """
        raise NotImplementedError()
