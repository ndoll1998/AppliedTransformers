import os
import json
import pandas as pd
# import pytorch and transformers
import torch
import transformers
# import base dataset
from base import BaseDataset
# utils
from itertools import combinations, product


class RelationExtractionDataset(BaseDataset):
    """ Base Dataset for the Relation Extraction Task """

    def __init__(self, input_ids, e1_e2_starts, relation_ids):
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, input_ids, e1_e2_starts, relation_ids)

    @property
    def num_relations(self):
        raise NotImplementedError()


""" SemEval2010 Task8 """

class SemEval2010Task8(RelationExtractionDataset):
    """ SemEval2010 Task8 Dataset
        Download: https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip
    """

    TRAIN_FILE = "SemEval2010-Task8/SemEval2010_task8_training/TRAIN_FILE.TXT"
    TEST_FILE = "SemEval2010-Task8/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    RELATIONS = [
        "Other",
        "Component-Whole(e2,e1)",       "Component-Whole(e1,e2)",
        "Instrument-Agency(e2,e1)",     "Instrument-Agency(e1,e2)",
        "Member-Collection(e2,e1)",     "Member-Collection(e1,e2)",
        "Cause-Effect(e2,e1)",          "Cause-Effect(e1,e2)",
        "Entity-Destination(e2,e1)",    "Entity-Destination(e1,e2)",
        "Content-Container(e2,e1)",     "Content-Container(e1,e2)",
        "Message-Topic(e2,e1)",         "Message-Topic(e1,e2)",
        "Product-Producer(e2,e1)",      "Product-Producer(e1,e2)",
        "Entity-Origin(e2,e1)",         "Entity-Origin(e1,e2)"
    ]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # build path to source file
        fname = SemEvalDataset.TRAIN_FILE if train else SemEvalDataset.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

        # build relation-to-index map
        rel2id = {rel: i for i, rel in enumerate(SemEvalDataset.RELATIONS)}

        # load data
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        # read examples
        input_ids, e1_e2_starts, relation_ids = [], [], []
        for sent_line, relation_line in zip(lines[::4], lines[1::4]):
            # get text
            sent = sent_line.split('\t')[1].strip()
            # clean up sentence
            assert sent[0] == sent[-1] == '"'
            sent = sent[1:-1]
            # mark entities and tokenize
            sent = sent.replace('<e1>', '[e1]').replace('</e1>', '[/e1]').replace('<e2>', '[e2]').replace('</e2>', '[/e2]')
            token_ids = tokenizer.encode(sent)[:seq_length]
            # check if entity is out of bounds
            if (tokenizer._entity1_token_id not in token_ids) or (tokenizer._entity2_token_id not in token_ids): 
                continue
            # get entity start positions
            entity_starts = (token_ids.index(tokenizer.entity1_token_id), token_ids.index(tokenizer.entity2_token_id))
            # read relation and get id
            relation = relation_line.strip()
            relation_id = rel2id[relation]
            # update lists
            input_ids.append(token_ids + [tokenizer.pad_token_id] * (seq_length - len(token_ids)))
            e1_e2_starts.append(entity_starts)
            relation_ids.append(rel2id[relation])
    
        # convert to tensors
        input_ids = torch.LongTensor(input_ids)
        e1_e2_starts = torch.LongTensor(e1_e2_starts)
        relation_ids = torch.LongTensor(relation_ids)
        # create dataset and dataloader
        RelationExtractionDataset.__init__(self, input_ids, e1_e2_start, relation_ids)

    @property
    def num_relations(self):
        return len(SemEvalDataset.RELATIONS)


""" German Yelp Dataset """

class GermanYelpRelations(RelationExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    RELATIONS = ["False", "True"]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpRelations.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpRelations.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
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
        return len(GermanYelpRelations.RELATIONS)


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

""" Smartdata Corpus """

class SmartdataCorpus(RelationExtractionDataset):
    """ German Smartdata Corpus
        Download: https://github.com/DFKI-NLP/smartdata-corpus/tree/master/v2_20190802
    """

    TRAIN_FILE = "SmartdataCorpus/train.json"
    TEST_FILE = "SmartdataCorpus/test.json"

    RELATIONS = ["Acquisition", "Insolvency", "Layoffs", "Merger", "OrganizationLeadership", "SpinOff", "Strike"]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # build full path 
        fname = SmartdataCorpus.TRAIN_FILE if train else SmartdataCorpus.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)
        # read data
        with open(fpath, 'r', encoding='utf-8') as f:
            json_dumps = f.read().split('}\n{')
            json_dumps = [('' if i == 0 else '{') +  s + ('' if i == len(json_dumps) - 1 else '}') for i, s in enumerate(json_dumps)]
            documents = [json.loads(dump) for dump in json_dumps][1:]

        # build relation-to-index map
        rel2id = {rel: i for i, rel in enumerate(SmartdataCorpus.RELATIONS)}

        input_ids, e1_e2_starts, relation_labels = [], [], []
        # process data
        for doc in documents:
            for sent in doc['sentences']['array']:
                # get sentence string
                begin, end = sent['span']['start'], sent['span']['end']
                sent_string = doc['text']['string'][begin:end]
                off = -begin

                for rel in sent['relationMentions']['array']:
                    rel_type = rel['name']

                    # check type
                    if rel_type not in rel2id:
                        continue
                    
                    types = ['organization-company', 'person', 'org-position', 'trigger', 'organization']
                    args = tuple(arg for arg in rel['args']['array'] if arg['conceptMention']['type'] in types)
                    # create bi-relations from n-ary relations
                    for argA, argB in combinations(args, 2):
                        # sort arguments
                        argA, argB = argA['conceptMention'], argB['conceptMention']
                        argA, argB = sorted((argA, argB), key=lambda a: a['span']['start'])
                        # mark in text
                        marked_sent = (sent_string[:off + argA['span']['start']] + \
                            "[e1]" + sent_string[off + argA['span']['start']:off + argA['span']['end']] + "[/e1]" + \
                            sent_string[off + argA['span']['end']:off + argB['span']['start']] + \
                            "[e2]" + sent_string[off + argB['span']['start']:off + argB['span']['end']] + "[/e2]" + \
                            sent_string[off + argB['span']['end']:]).replace('\n', ' ').strip()
                        # tokenize sentence
                        token_ids = tokenizer.encode(marked_sent)[:seq_length]
                        token_ids += [tokenizer.pad_token_id] * (seq_length - len(token_ids))
                        # find entity starts
                        if (tokenizer._entity1_token_id not in token_ids) or (tokenizer._entity2_token_id not in token_ids): 
                            continue
                        entity_starts = (token_ids.index(tokenizer.entity1_token_id), token_ids.index(tokenizer.entity2_token_id))
                        # add to lists
                        input_ids.append(token_ids)
                        e1_e2_starts.append(entity_starts)
                        relation_labels.append(rel2id[rel_type])

        # convert to tensors
        input_ids = torch.LongTensor(input_ids)
        e1_e2_starts = torch.LongTensor(e1_e2_starts)
        labels = torch.LongTensor(relation_labels)
        # initialize dataset
        RelationExtractionDataset.__init__(self, input_ids, e1_e2_starts, labels)

    @property
    def num_relations(self):
        return len(SmartdataCorpus.RELATIONS)
