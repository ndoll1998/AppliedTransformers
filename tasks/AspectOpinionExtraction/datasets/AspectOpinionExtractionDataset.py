# import pytorch
import torch
# import base dataset
from base import BaseDataset

class AspectOpinionExtractionDataset(BaseDataset):
    """ Base Dataset for the Aspect-Opinion Extraction Task """

    def __init__(self, input_ids, labels_a, labels_o):
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, input_ids, labels_a, labels_o)
