import os
import json
# import pytorch and transformers
import torch
import transformers
# import base dataset
from .RelationExtractionDataset import RelationExtractionDataset
# utils
from itertools import combinations

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
