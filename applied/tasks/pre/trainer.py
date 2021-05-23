  
from applied.core.trainer import Trainer as BaseTrainer
from applied.core.metrics import MetricCollection, Losses, MicroF1Score, MacroF1Score
# import model and dataset
from .models.base import PretrainModel
from .datasets.base import PretrainDataset

MicroF1 = type("MicroF1", (MicroF1Score,), {'idx': 0, 'ignore_label': -1})
MacroF1 = type("MacroF1", (MacroF1Score,), {'idx': 0, 'ignore_label': -1})

class Trainer(BaseTrainer):
    # model and dataset type
    BASE_MODEL_TYPE = PretrainModel
    BASE_DATASET_TYPE = PretrainDataset
    # metrics type
    METRIC_TYPE = MetricCollection[Losses, 
        MetricCollection[MicroF1, MacroF1].share_axes(),
    ]
