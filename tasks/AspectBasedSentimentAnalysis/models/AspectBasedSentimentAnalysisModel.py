# import base model
from core.Model import BaseModel

class AspectBasedSentimentAnalysisModel(BaseModel):
    """ Base model for the aspects based sentiment analysis task """

    def prepare(self, input_ids:list, aspect_terms:list, labels:list, tokenizer) -> list:
        """ Prepare a dataset item for the model. """
        return [(text, aspect_terms, labels)]
