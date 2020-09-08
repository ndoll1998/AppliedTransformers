import os
import pandas as pd
# import base dataset
from .RelationExtractionDataset import RelationExtractionDataset
# utils
from itertools import product

class __GermanYelp_Base(RelationExtractionDataset):
    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

class GermanYelp_Polarity(__GermanYelp_Base):
    # list of relation types
    RELATIONS = ["positive", "negative"]

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelp_Polarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelp_Polarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # remove all annotations that are not a relation
        annotations.dropna(inplace=True)
        # separate training and testing set
        n_train_samples = int(len(annotations) * 0.8)

        for k, row in enumerate(annotations.itertuples()):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # yield item
            yield sentences[row.SentenceID], eval(row.Aspect), eval(row.Opinion), row.Sentiment


class GermanYelp_Linking(__GermanYelp_Base):
    # list of relation types
    RELATIONS = ["False", "True"]

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelp_Linking.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelp_Linking.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
            for t1, t2 in product(range(len(aspects)), range(len(opinions))):
                # get entities mark th
                aspect, opinion = aspects[t1], opinions[t2]
                # get label
                label = GermanYelp_Linking.RELATIONS[int((t1, t2) in relations)]
                # yield features
                yield sent, aspect, opinion, label


class GermanYelp_LinkingAndPolarity(__GermanYelp_Base):
    # list of relation types
    RELATIONS = ["none", "positive", "negative"]

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelp_LinkingAndPolarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelp_LinkingAndPolarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
            for t1, t2 in product(range(len(aspects)), range(len(opinions))):
                # get entities mark th
                aspect, opinion = aspects[t1], opinions[t2]
                # get label
                label = sentiments[relations.index((t1, t2))] if (t1, t2) in relations else 'none'
                # yield features
                yield sent, aspect, opinion, label
