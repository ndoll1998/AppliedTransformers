# import numpy
import numpy as np
# import base predictor
from core.Predictor import BasePredictor
# import base model and dataset type
from .models import RelationExtractionModel
from .datasets import RelationExtractionDataset

class RelationExtractionPredictor(BasePredictor):

    BASE_MODEL_TYPE = RelationExtractionModel
    BASE_DATASET_TYPE = RelationExtractionDataset

    def postprocess(self, logits, *additionals):
        # get numpy array of all posible labels
        labels = np.array(self.dataset_type.RELATIONS)
        # get predicted label
        pred = logits.max(dim=-1)[1].cpu().numpy()
        label = labels[pred]
        # return
        return label[0]
