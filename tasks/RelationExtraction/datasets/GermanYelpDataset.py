import os
import pandas as pd
# import pytorch and transformers
import torch
import transformers
# import base dataset
from .RelationExtractionDataset import RelationExtractionDataset
# utils
from itertools import product


class GermanYelpRelation(RelationExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    RELATIONS = ["False", "True"]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpRelation.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpRelation.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)

        input_ids, e1_e2_starts, relation_ids = [], [], []
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
                if (a['Opinion'] not in aspects) and (a['Opinion'] == a['Opinion']):
                    opinions.append(a['Opinion'])
                if (a['Aspect'] == a['Aspect']) and (a['Opinion'] == a['Opinion']):
                    aspect_id, opinion_id = aspects.index(a['Aspect']), opinions.index(a['Opinion'])
                    relations.append((aspect_id, opinion_id))

            # create aspect and opinion entities
            aspects = list(map(lambda aspect: ('[e1]', '[/e1]', *eval(aspect)), aspects))
            opinions = list(map(lambda opinion: ('[e2]', '[/e2]', *eval(opinion)), opinions))
            # mark entity
            mark = lambda e: e[0] + sent[e[2]:e[3]] + e[1]

            # add all positive labels
            for t1, t2 in product(range(len(aspects)), range(len(opinions))):
                # get entities
                aspect, opinion = aspects[t1], opinions[t2]
                e1, e2 = sorted([aspect, opinion], key=lambda e: e[2])
                # mark entities in sentence
                marked_sent = sent[:e1[2]] + mark(e1) + sent[e1[3]:e2[2]] + mark(e2) + sent[e2[3]:]
                token_ids = tokenizer.encode(marked_sent)[:seq_length]
                # check if entity is out of bounds
                if (tokenizer._entity1_token_id not in token_ids) or (tokenizer._entity2_token_id not in token_ids): 
                    continue
                # get entity start positions
                entity_starts = (token_ids.index(tokenizer.entity1_token_id), token_ids.index(tokenizer.entity2_token_id))
                # add to lists
                input_ids.append(token_ids + [tokenizer.pad_token_id] * (seq_length - len(token_ids)))
                e1_e2_starts.append(entity_starts)
                relation_ids.append(int((t1, t2) in relations))

        # convert to tensors
        input_ids = torch.LongTensor(input_ids)
        e1_e2_starts = torch.LongTensor(e1_e2_starts)
        relation_ids = torch.LongTensor(relation_ids)
        # create dataset and dataloader
        RelationExtractionDataset.__init__(self, input_ids, e1_e2_starts, relation_ids)

    @property
    def num_relations(self):
        return len(GermanYelpRelation.RELATIONS)


class GermanYelpPolarity(RelationExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    RELATIONS = ["none", "positive", "negative"]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):
        # build relation-to-index map
        rel2id = {rel: i for i, rel in enumerate(GermanYelpPolarity.RELATIONS)}

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpPolarity.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpPolarity.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)

        input_ids, e1_e2_starts, relation_ids = [], [], []
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
                if (a['Opinion'] not in aspects) and (a['Opinion'] == a['Opinion']):
                    opinions.append(a['Opinion'])
                if (a['Aspect'] == a['Aspect']) and (a['Opinion'] == a['Opinion']):
                    aspect_id, opinion_id = aspects.index(a['Aspect']), opinions.index(a['Opinion'])
                    relations.append((aspect_id, opinion_id))
                    sentiments.append(a['Sentiment'])

            # create aspect and opinion entities
            aspects = list(map(lambda aspect: ('[e1]', '[/e1]', *eval(aspect)), aspects))
            opinions = list(map(lambda opinion: ('[e2]', '[/e2]', *eval(opinion)), opinions))
            # mark entity
            mark = lambda e: e[0] + sent[e[2]:e[3]] + e[1]

            # add all positive labels
            for t1, t2 in product(range(len(aspects)), range(len(opinions))):
                # get entities
                aspect, opinion = aspects[t1], opinions[t2]
                e1, e2 = sorted([aspect, opinion], key=lambda e: e[2])
                # mark entities in sentence
                marked_sent = sent[:e1[2]] + mark(e1) + sent[e1[3]:e2[2]] + mark(e2) + sent[e2[3]:]
                token_ids = tokenizer.encode(marked_sent)[:seq_length]
                # check if entity is out of bounds
                if (tokenizer._entity1_token_id not in token_ids) or (tokenizer._entity2_token_id not in token_ids): 
                    continue
                # get entity start positions
                entity_starts = (token_ids.index(tokenizer.entity1_token_id), token_ids.index(tokenizer.entity2_token_id))
                # add to lists
                input_ids.append(token_ids + [tokenizer.pad_token_id] * (seq_length - len(token_ids)))
                e1_e2_starts.append(entity_starts)
                if (t1, t2) in relations:
                    i = relations.index((t1, t2))
                    sentiment = sentiments[i]
                    relation_ids.append(rel2id[sentiment])
                else:
                    relation_ids.append(rel2id['none'])

        # convert to tensors
        input_ids = torch.LongTensor(input_ids)
        e1_e2_starts = torch.LongTensor(e1_e2_starts)
        relation_ids = torch.LongTensor(relation_ids)
        # create dataset and dataloader
        RelationExtractionDataset.__init__(self, input_ids, e1_e2_starts, relation_ids)

    @property
    def num_relations(self):
        return len(GermanYelpPolarity.RELATIONS)
