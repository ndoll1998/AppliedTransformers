import os
# import pytorch
import torch
# import numpy and pandas
import numpy as np
import pandas as pd
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset

class GermanYelpSentiment(EntityClassificationDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    LABELS = ["positive", "negative"]

    def yield_item_features(self, train:bool, data_base_dir:str) -> iter:
        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpSentiment.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpSentiment.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)

        for sent_id in annotations['SentenceID'].unique():
            # only load train or test data, not both
            if ((sent_id < n_train_samples) and not train) or ((sent_id >= n_train_samples) and train):
                continue
            # get sentence
            sent = sentences[sent_id]
            # get all annotations of the current sentence
            sent_annotations = annotations[annotations['SentenceID'] == sent_id]
            opinions = sent_annotations['Opinion']
            opinions = sent_annotations[opinions == opinions]
            opinions, sentiments = opinions['Opinion'].values, opinions['Sentiment'].values
            # remove double opinions
            opinions, unique_idx = np.unique(opinions, return_index=True)
            sentiments = sentiments[unique_idx]
            # convert strings to tuples
            opinions = list(map(eval, opinions))

            # yield features for this item
            yield sent, opinions, sentiments