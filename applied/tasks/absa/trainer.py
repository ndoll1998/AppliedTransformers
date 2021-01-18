from applied.core.trainer import Trainer as BaseTrainer
from applied.core.metrics import MetricCollection, Losses, MicroF1Score, MacroF1Score
# import model and dataset
from .models.base import ABSA_Model
from .datasets.base import ABSA_Dataset

class Trainer(BaseTrainer):
    # model and dataset type
    BASE_MODEL_TYPE = ABSA_Model
    BASE_DATASET_TYPE = ABSA_Dataset
    # metrics type
    METRIC_TYPE = MetricCollection[Losses,
        MetricCollection[
            MicroF1Score, 
            MacroF1Score
        ].share_axes()
    ]
