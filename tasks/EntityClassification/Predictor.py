# import numpy
import numpy as np
# import base predictor
from core.Predictor import BasePredictor
# import base model and dataset type
from .models import EntityClassificationModel
from .datasets import EntityClassificationDataset

class EntityClassificationPredictor(BasePredictor):

    BASE_MODEL_TYPE = EntityClassificationModel
    BASE_DATASET_TYPE = EntityClassificationDataset

    def postprocess(self, logits, *additionals):
        # get numpy array of all posible labels
        labels = np.array(self.dataset_type.LABELS)
        # get predicted label
        pred = logits.max(dim=-1)[1].cpu().numpy()
        label = labels[pred]
        # return
        return label
