from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class AOEx_DatasetItem(DatasetItem):
    sentence:str
    aspect_spans:tuple
    opinion_spans:tuple

class AOEx_Dataset(Dataset): pass
