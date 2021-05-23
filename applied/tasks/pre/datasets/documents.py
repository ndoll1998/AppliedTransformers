import os
from random import shuffle
from .base import PretrainDataset, PretrainDatasetItem

class Documents(PretrainDataset):
    """ Dataset that generates items from a set
        of txt-files with the following layout:
            - each line represents single sentence
            - documents are seperated by an empty line
    """

    def __init__(self, train_files:list, eval_files:list, *args, **kwargs):
        # initialize the base dataset and save the folder
        # containing the document files
        PretrainDataset.__init__(self, *args, **kwargs)
        self.__train_files = eval_files # TODO: use train files
        self.__eval_files = eval_files

    def yield_items(self, files:list) -> iter:
        # loop over all document files
        for fname in files:
            # build full path to document file
            fpath = os.path.join(self.data_base_dir, fname)
            # read documents
            with open(fpath, 'r', encoding='utf-8') as f:
                docs = f.read().split('\n\n')
                docs = [doc.strip().split('\n') for doc in docs if len(doc.strip()) > 0]
            # shuffle and yield documents
            shuffle(docs)
            yield PretrainDatasetItem(documents=docs)

    # yield train and eval items
    yield_train_items = lambda self: self.yield_items(self.__train_files)
    yield_eval_items = lambda self: self.yield_items(self.__eval_files)
    # number of dataset items
    n_train_items = lambda self: len(self.__train_files)
    n_eval_items = lambda self: len(self.__eval_files)
