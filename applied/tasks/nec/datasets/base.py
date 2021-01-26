from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class NEC_DatasetItem(DatasetItem):
    sentence:str
    entity_spans:tuple
    labels:tuple

class NEC_Dataset(Dataset): pass