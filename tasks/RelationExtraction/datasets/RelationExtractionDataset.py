import torch
# import base dataset
from base import BaseDataset

class RelationExtractionDataset(BaseDataset):
    """ Base Dataset for the Relation Extraction Task """

    def __init__(self, input_ids, e1_e2_starts, relation_ids):
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, input_ids, e1_e2_starts, relation_ids)

    @property
    def num_relations(self):
        raise NotImplementedError()