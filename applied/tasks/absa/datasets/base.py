from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class ABSA_DatasetItem(DatasetItem):
    sentence:str
    aspects:tuple
    labels:tuple

class ABSA_Dataset(Dataset): pass
