# use the same trainer as for the entity classification
from ..EntityClassification.Trainer import EntityClassificationTrainer
# import base model and dataset
from .models import AspectBasedSentimentAnalysisModel
from .datasets import AspectBasedSentimentAnalysisDataset

class AspectBasedSentimentAnalysisTrainer(EntityClassificationTrainer):

    # set base types for model and dataset
    BASE_MODEL_TYPE = AspectBasedSentimentAnalysisModel
    BASE_DATASET_TYPE = AspectBasedSentimentAnalysisDataset
