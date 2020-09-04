import os
# import numpy and pandas
import numpy as np
import pandas as pd
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset

class __GermanYelp_Base(EntityClassificationDataset):
    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"
    # entity labels
    LABELS = ["positive", "negative"]


class GermanYelp_OpinionPolarity(__GermanYelp_Base):

    def yield_item_features(self, train:bool, data_base_dir:str) -> iter:
        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelp_OpinionPolarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelp_OpinionPolarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
            opinions, polarities = opinions['Opinion'].values, opinions['Sentiment'].values
            # remove double opinions
            opinions, unique_idx = np.unique(opinions, return_index=True)
            polarities = polarities[unique_idx]
            # convert strings to tuples
            opinions = list(map(eval, opinions))

            # yield features for this item
            yield sent, opinions, polarities


class GermanYelp_AspectPolarity(__GermanYelp_Base):

    def yield_item_features(self, train:bool, data_base_dir:str ='./data') -> iter:
        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelp_OpinionPolarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelp_OpinionPolarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # remove all non-relation annotations
        annotations.dropna(inplace=True)
        unique_sent_ids = annotations['SentenceID'].unique()
        # separate training and testing set
        n_train_samples = int(len(unique_sent_ids) * 0.8)

        for k, sent_id in enumerate(unique_sent_ids):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # get sentence and it's annotations
            sent = sentences[sent_id]
            sent_annotations = annotations[annotations['SentenceID'] == sent_id]
            # get aspects and sentiments
            aspects, polarities = [], []
            # group annotations by aspects
            for aspect in sent_annotations['Aspect'].unique():
                # get all annotations with the same aspect
                aspect_annotations = sent_annotations[sent_annotations['Aspect'] == aspect]
                # get polarity by majority
                n_positives = (aspect_annotations['Sentiment'].values == GermanYelp_AspectPolarity.LABELS[0]).sum()
                is_positive = n_positives >= len(aspect_annotations.index) // 2
                polarity = GermanYelp_OpinionPolarity.LABELS[1 - int(is_positive)]
                # add to lists
                aspects.append(eval(aspect))
                polarities.append(polarity)

            # yield item
            yield sent, aspects, polarities
