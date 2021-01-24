import os
import numpy as np
import pandas as pd
from .base import EC_Dataset, EC_DatasetItem


class __GermYelp(EC_Dataset):
    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"
    # entity labels
    LABELS = ["positive", "negative"]

    # yield train and eval items
    yield_train_items = lambda self: self.yield_items(train=True)
    yield_eval_items = lambda self: self.yield_items(train=False)


class GermYelp_OpinionPolarity(__GermYelp):
    def yield_items(self, train:bool) -> iter:
        # build file paths
        annotations_fpath = os.path.join(self.data_base_dir, GermYelp_OpinionPolarity.ANNOTATIONS_FILE)
        sentences_fpath = os.path.join(self.data_base_dir, GermYelp_OpinionPolarity.SENTENCES_FILE)
        # load annotations and sentences
        annotations = pd.read_csv(annotations_fpath, sep="\t", index_col=0)
        with open(sentences_fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
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
            yield EC_DatasetItem(
                sentence=sent, 
                entity_spans=opinions, 
                labels=[GermYelp_OpinionPolarity.LABELS.index(p) for p in polarities]
            )

class GermYelp_AspectPolarity(__GermYelp):
    def yield_items(self, train:bool) -> iter:
        # build file paths
        annotations_fpath = os.path.join(self.data_base_dir, GermYelp_AspectPolarity.ANNOTATIONS_FILE)
        sentences_fpath = os.path.join(self.data_base_dir, GermYelp_AspectPolarity.SENTENCES_FILE)
        # load annotations and sentences
        annotations = pd.read_csv(annotations_fpath, sep="\t", index_col=0)
        with open(sentences_fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
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
                n_positives = (aspect_annotations['Sentiment'].values == GermYelp_AspectPolarity.LABELS[0]).sum()
                n_negatives = len(aspect_annotations.index) - n_positives
                is_positive = (n_positives >= n_negatives)
                polarity = 1 - int(is_positive)
                # add to lists
                aspects.append(eval(aspect))
                polarities.append(polarity)

            # yield item
            yield EC_DatasetItem(
                sentence=sent,
                entity_spans=aspects,
                labels=polarities
            )
