from applied.core.dataset import Dataset, DatasetItem
from dataclasses import dataclass

@dataclass(frozen=True)
class PretrainDatasetItem(DatasetItem):
    # documents are lists of sentences
    documents:list

class PretrainDataset(Dataset): pass
