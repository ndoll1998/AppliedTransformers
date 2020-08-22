# import torch
import torch
# import base dataset
from base import BaseDataset

class EntityClassificationDataset(BaseDataset):
    """ Base Dataset for the Entity Classification Task """

    def __init__(self, input_ids, entity_starts, labels):
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, input_ids, entity_starts, labels)

    @property
    def num_labels(self):
        raise NotImplementedError()

