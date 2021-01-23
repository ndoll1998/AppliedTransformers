import os
import pandas as pd
import itertools as it
# import base dataset
from .base import RelExDataset, RelExDatasetItem


class __GermYelp(RelExDataset):
    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

class GermYelp_Polarity(__GermYelp):
    LABELS = ["positive", "negative"]

    # yield items
    yield_train_items = lambda self: self.yield_items(train=True)
    yield_eval_items = lambda self: self.yield_items(train=False)

    def yield_items(self, train:bool) -> iter:
        # build filepaths
        annotations_fpath = os.path.join(self.data_base_dir, GermYelp_Polarity.ANNOTATIONS_FILE)
        sentences_fpath = os.path.join(self.data_base_dir, GermYelp_Polarity.SENTENCES_FILE)
        # load annotations and sentences
        annotations = pd.read_csv(annotations_fpath, sep="\t", index_col=0)
        with open(sentences_fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
        # remove all annotations that are not a relation
        annotations.dropna(inplace=True)
        # separate training and testing set
        n_train_samples = int(len(annotations) * 0.8)

        for k, row in enumerate(annotations.itertuples()):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # yield item
            yield RelExDatasetItem(
                sentence=sentences[row.SentenceID], 
                source_entity_span=eval(row.Aspect), 
                target_entity_span=eval(row.Opinion), 
                relation_type=GermYelp_Polarity.LABELS.index(row.Sentiment.lower())
            )

class GermYelp_Linking(__GermYelp):
    LABELS = ["false", "true"]

    # yield items
    yield_train_items = lambda self: self.yield_items(train=True)
    yield_eval_items = lambda self: self.yield_items(train=False)

    def yield_items(self, train:bool) -> iter:
        # build filepaths
        annotations_fpath = os.path.join(self.data_base_dir, GermYelp_Linking.ANNOTATIONS_FILE)
        sentences_fpath = os.path.join(self.data_base_dir, GermYelp_Linking.SENTENCES_FILE)
        # load annotations and sentences
        annotations = pd.read_csv(annotations_fpath, sep="\t", index_col=0)
        with open(sentences_fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
        # separate training and testing set
        n_train_samples = int(len(sentences) * 0.8)

        for sent_id in annotations['SentenceID'].unique():
            # only load train or test data, not both
            if ((sent_id < n_train_samples) and not train) or ((sent_id >= n_train_samples) and train):
                continue
            # get sentence
            sent = sentences[sent_id]
            # get all annotations of the current sentence
            sent_annotations = annotations[annotations['SentenceID'] == sent_id]
            aspects, opinions, relations = [], [], []
            # gather all aspects, opinions and relations in the sentence
            for row in sent_annotations.itertuples():
                if (row.Aspect not in aspects) and (row.Aspect == row.Aspect):
                    aspects.append(row.Aspect)
                if (row.Opinion not in opinions) and (row.Opinion == row.Opinion):
                    opinions.append(row.Opinion)
                if (row.Aspect == row.Aspect) and (row.Opinion == row.Opinion):
                    aspect_id, opinion_id = aspects.index(row.Aspect), opinions.index(row.Opinion)
                    relations.append((aspect_id, opinion_id))

            # convert aspect- and opinion-span-strings to tuples
            aspects = list(map(eval, aspects))
            opinions = list(map(eval, opinions))

            # create relations between all aspects and opinions
            # invalid relations have the label "none" assigned to them
            for t1, t2 in it.product(range(len(aspects)), range(len(opinions))):
                # get entities mark th
                aspect, opinion = aspects[t1], opinions[t2]
                # get label
                label = int((t1, t2) in relations)
                # yield features
                yield RelExDatasetItem(
                    sentence=sent, 
                    source_entity_span=aspect, 
                    target_entity_span=opinion, 
                    relation_type=label
                )


class GermYelp_LinkingAndPolarity(__GermYelp):
    LABELS = ["none", "positive", "negative"]

    # yield items
    yield_train_items = lambda self: self.yield_items(train=True)
    yield_eval_items = lambda self: self.yield_items(train=False)

    def yield_items(self, train:bool) -> iter:
        # build filepaths
        annotations_fpath = os.path.join(self.data_base_dir, GermYelp_LinkingAndPolarity.ANNOTATIONS_FILE)
        sentences_fpath = os.path.join(self.data_base_dir, GermYelp_LinkingAndPolarity.SENTENCES_FILE)
        # load annotations and sentences
        annotations = pd.read_csv(annotations_fpath, sep="\t", index_col=0)
        with open(sentences_fpath, 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
        # separate training and testing set
        n_train_samples = int(len(sentences) * 0.8)

        for sent_id in annotations['SentenceID'].unique():
            # only load train or test data, not both
            if ((sent_id < n_train_samples) and not train) or ((sent_id >= n_train_samples) and train):
                continue
            # get sentence
            sent = sentences[sent_id]
            # get all annotations of the current sentence
            sent_annotations = annotations[annotations['SentenceID'] == sent_id]
            aspects, opinions, relations, sentiments = [], [], [], []
            # gather all aspects, opinions and relations in the sentence
            for row in sent_annotations.itertuples():
                if (row.Aspect not in aspects) and (row.Aspect == row.Aspect):
                    aspects.append(row.Aspect)
                if (row.Opinion not in opinions) and (row.Opinion == row.Opinion):
                    opinions.append(row.Opinion)
                if (row.Aspect == row.Aspect) and (row.Opinion == row.Opinion):
                    aspect_id, opinion_id = aspects.index(row.Aspect), opinions.index(row.Opinion)
                    relations.append((aspect_id, opinion_id))
                    sentiments.append(row.Sentiment)

            # convert aspect- and opinion-span-strings to tuples
            aspects = list(map(eval, aspects))
            opinions = list(map(eval, opinions))

            # create relations between all aspects and opinions
            # invalid relations have the label "none" assigned to them
            for t1, t2 in it.product(range(len(aspects)), range(len(opinions))):
                # get entities mark th
                aspect, opinion = aspects[t1], opinions[t2]
                # get label
                label = sentiments[relations.index((t1, t2))].lower() if (t1, t2) in relations else 'none'
                label = GermYelp_LinkingAndPolarity.LABELS.index(label)
                # yield features
                yield RelExDatasetItem(
                    sentence=sent, 
                    source_entity_span=aspect, 
                    target_entity_span=opinion, 
                    relation_type=label
                )