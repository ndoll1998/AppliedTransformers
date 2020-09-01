import os
import pandas as pd
# import base dataset
from .RelationExtractionDataset import RelationExtractionDataset
# utils
from itertools import product


class GermanYelpRelation(RelationExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    RELATIONS = ["False", "True"]

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpRelation.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpRelation.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
            for _, a in sent_annotations.iterrows():
                if (a['Aspect'] not in aspects) and (a['Aspect'] == a['Aspect']):
                    aspects.append(a['Aspect'])
                if (a['Opinion'] not in opinions) and (a['Opinion'] == a['Opinion']):
                    opinions.append(a['Opinion'])
                if (a['Aspect'] == a['Aspect']) and (a['Opinion'] == a['Opinion']):
                    aspect_id, opinion_id = aspects.index(a['Aspect']), opinions.index(a['Opinion'])
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
                label = GermanYelpRelation.RELATIONS[int((t1, t2) in relations)]
                # yield features
                yield sent, aspect, opinion, label


class GermanYelpPolarity(RelationExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    RELATIONS = ["none", "positive", "negative"]

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpPolarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpPolarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
            for _, a in sent_annotations.iterrows():
                if (a['Aspect'] not in aspects) and (a['Aspect'] == a['Aspect']):
                    aspects.append(a['Aspect'])
                if (a['Opinion'] not in opinions) and (a['Opinion'] == a['Opinion']):
                    opinions.append(a['Opinion'])
                if (a['Aspect'] == a['Aspect']) and (a['Opinion'] == a['Opinion']):
                    aspect_id, opinion_id = aspects.index(a['Aspect']), opinions.index(a['Opinion'])
                    relations.append((aspect_id, opinion_id))
                    sentiments.append(a['Sentiment'])

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
