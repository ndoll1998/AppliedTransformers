from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class EC_DatasetItem(DatasetItem):
    sentence:str
    entity_spans:tuple
    labels:tuple

class EC_Dataset(Dataset): pass