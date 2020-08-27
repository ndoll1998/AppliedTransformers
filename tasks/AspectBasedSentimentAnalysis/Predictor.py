# import numpy
import numpy as np
# import base predictor
from core.Predictor import BasePredictor
# import base model and dataset type
from .models import AspectBasedSentimentAnalysisModel
from .datasets import AspectBasedSentimentAnalysisDataset

class AspectBasedSentimentAnalysisPredictor(BasePredictor):

    BASE_MODEL_TYPE = AspectBasedSentimentAnalysisModel
    BASE_DATASET_TYPE = AspectBasedSentimentAnalysisDataset

    def postprocess(self, logits, *additionals):
        # get numpy array of all posible labels
        labels = np.array(self.dataset_type.LABELS)
        # get predicted label
        pred = logits.max(dim=-1)[1].cpu().numpy()
        label = labels[pred]
        # return
        return label
