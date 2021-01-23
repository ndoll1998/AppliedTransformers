from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class RelExDatasetItem(DatasetItem):
    sentence:str
    source_entity_span:tuple
    target_entity_span:tuple
    relation_type:str

class RelExDataset(Dataset): pass