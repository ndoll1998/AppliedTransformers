from applied.core.trainer import Trainer as BaseTrainer
from applied.core.metrics import MetricCollection, Losses, MicroF1Score, MacroF1Score
# import model and dataset
from .models.base import AOEx_Model
from .datasets.base import AOEx_Dataset

# F1-Score Metrics for aspects and opinions
Aspect_MicroF1 = type("Aspect_MicroF1", (MicroF1Score,), {'idx': 0, 'ignore_label': -1})
Aspect_MacroF1 = type("Aspect_MacroF1", (MacroF1Score,), {'idx': 0, 'ignore_label': -1})
Opinion_MicroF1 = type("Opinion_MicroF1", (MicroF1Score,), {'idx': 1, 'ignore_label': -1})
Opinion_MacroF1 = type("Opinion_MacroF1", (MacroF1Score,), {'idx': 1, 'ignore_label': -1})

class Trainer(BaseTrainer):
    # model and dataset type
    BASE_MODEL_TYPE = AOEx_Model
    BASE_DATASET_TYPE = AOEx_Dataset
    # metrics type
    METRIC_TYPE = MetricCollection[Losses, 
        MetricCollection[Aspect_MicroF1, Opinion_MicroF1].share_axes(),
        MetricCollection[Aspect_MacroF1, Opinion_MacroF1].share_axes()
    ]
