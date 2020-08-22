import os
# import pytorch and transformers
import torch
import transformers
# import base dataset
from .RelationExtractionDataset import RelationExtractionDataset

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
        fname = SemEval2010Task8.TRAIN_FILE if train else SemEval2010Task8.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

        # build relation-to-index map
        rel2id = {rel: i for i, rel in enumerate(SemEval2010Task8.RELATIONS)}

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
        RelationExtractionDataset.__init__(self, input_ids, e1_e2_starts, relation_ids)

    @property
    def num_relations(self):
        return len(SemEval2010Task8.RELATIONS)