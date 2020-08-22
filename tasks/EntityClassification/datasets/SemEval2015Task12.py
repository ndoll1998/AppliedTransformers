import os
# import pytorch and transformers
import torch
import transformers
# import xml parser
import xml.etree.ElementTree as ET
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset
# utils
from utils import match_2d_shape

class SemEval2015Task12_AspectSentiment(EntityClassificationDataset):
    """ Dataset for the SemEval2014 Task4 data for Aspect-based Sentiment Analysis
        Download: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
    """

    TRAIN_FILE = "SemEval2015-Task12/ABSA-15_Restaurants_Train_Final.xml"
    TEST_FILE = "SemEval2015-Task12/ABSA15_Restaurants_Test.xml"

    LABELS = ['positive', 'neutral', 'negative']

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):
        # create label-to-id map
        label2id = {l:i for i, l in enumerate(SemEval2015Task12_AspectSentiment.LABELS)}

        # build full paths to files
        fname = SemEval2015Task12_AspectSentiment.TRAIN_FILE if train else SemEval2015Task12_AspectSentiment.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

        # parse xml file
        tree = ET.parse(fpath)
        root = tree.getroot()
        
        all_input_ids, all_entity_starts, all_labels = [], [], []
        # parse all reviews
        for review in root:
            for sent in review[0].findall('sentence'):
                # get sentence
                text = sent.find('text').text
                # find opinions
                opinions = sent.find('Opinions')
                if opinions is None:
                    continue
                # get aspects and sentiments
                aspects = [(int(o.attrib['from']), int(o.attrib['to'])) for o in opinions]
                sentiments = [o.attrib['polarity'] for o in opinions]
                # remove unvalids - no aspect target
                sentiments = [s for s, (b, e) in zip(sentiments, aspects) if b < e]
                aspects = [(b, e) for (b, e) in aspects if b < e]
                # sort aspects
                sort_idx = sorted(range(len(aspects)), key=lambda i: aspects[i][0])
                aspects = [aspects[i] for i in sort_idx]
                sentiments = [sentiments[i] for i in sort_idx]
                # mark aspects in sentence
                for (b, e) in aspects[::-1]:
                    text = text[:b] + '[e]' + text[b:e] + '[/e]' + text[e:]
                # encode sentence
                input_ids = tokenizer.encode(text)[:seq_length]
                # find all entity starts
                entity_starts = [i for i, t in enumerate(input_ids) if t == tokenizer.entity_token_id]
                # no entities in bounds
                if len(entity_starts) == 0:
                    continue
                # get labels
                labels = [label2id[l] for l in sentiments[:len(entity_starts)]]
                # add to lists
                all_input_ids.append(input_ids + [tokenizer.pad_token_id] * (seq_length - len(input_ids)))
                all_entity_starts.append(entity_starts)
                all_labels.append(labels)

        n = len(all_input_ids)
        m = max((len(labels) for labels in all_labels))
        # convert to tensors
        input_ids = torch.LongTensor(all_input_ids)
        entity_starts = torch.LongTensor(match_2d_shape(all_entity_starts, (n, m), fill_val=-1))
        labels = torch.LongTensor(match_2d_shape(all_labels, (n, m), fill_val=-1))
        # initialize dataset
        EntityClassificationDataset.__init__(self, input_ids, entity_starts, labels)

    @property
    def num_labels(self):
        return len(SemEval2015Task12_AspectSentiment.LABELS)

class SemEval2015Task12_OpinionSentiment(EntityClassificationDataset):
    """ Dataset for the SemEval2014 Task4 data for Opinion-based Sentiment Analysis
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """

    TRAIN_FILE = "SemEval2015-Task12/sentence_res15_op.txt"
    TEST_FILE = "SemEval2015-Task12/sentence_restest15_op.txt"

    LABELS = ['+1', '-1']

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):
        # create label-to-id map
        label2id = {l:i for i, l in enumerate(SemEval2015Task12_OpinionSentiment.LABELS)}

        # build full paths to files
        fname = SemEval2015Task12_OpinionSentiment.TRAIN_FILE if train else SemEval2015Task12_OpinionSentiment.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

        # load file content
        with open(fpath, 'r', encoding='utf-8') as f:
            all_sents_opinions = f.read().split('\n')

        all_input_ids, all_entity_starts, all_labels = [], [], []
        # preprocess data
        for sent_opinions in all_sents_opinions:
            # no opinions
            if '##' not in sent_opinions:
                continue
            # separate sentence from opinions
            sent, opinions = sent_opinions.split('##')
            # get aspects and opinions
            opinions = [o.strip() for o in opinions.split(',')] if len(opinions) > 0 else []
            opinions, sentiments = zip(*[(o[:-2].strip(), o[-2:]) for o in opinions])
            # find opinions in sentence
            opinion_pos = [sent.find(o) for o in opinions]
            sort_idx = sorted(range(len(opinions)), key=lambda i: opinion_pos[i])
            # mark opinions in sentence
            for i, o in zip(opinion_pos[::-1], opinions[::-1]):
                sent = sent[:i] + '[e]' + o + '[/e]' + sent[i + len(o):]
            # encode sentence
            input_ids = tokenizer.encode(sent)[:seq_length]
            # find all entity starts
            entity_starts = [i for i, t in enumerate(input_ids) if t == tokenizer.entity_token_id]
            # no entities in bounds
            if len(entity_starts) == 0:
                continue
            # sort sentiments to match the order of entity-starts and get labels
            sentiments = [sentiments[i] for i in sort_idx]
            labels = [label2id[l] for l in sentiments[:len(entity_starts)]]
            # add to lists
            all_input_ids.append(input_ids + [tokenizer.pad_token_id] * (seq_length - len(input_ids)))
            all_entity_starts.append(entity_starts)
            all_labels.append(labels)

        n = len(all_input_ids)
        m = max((len(labels) for labels in all_labels))
        # convert to tensors
        input_ids = torch.LongTensor(all_input_ids)
        entity_starts = torch.LongTensor(match_2d_shape(all_entity_starts, (n, m), fill_val=-1))
        labels = torch.LongTensor(match_2d_shape(all_labels, (n, m), fill_val=-1))
        # initialize dataset
        EntityClassificationDataset.__init__(self, input_ids, entity_starts, labels)

    @property
    def num_labels(self):
        return len(SemEval2015Task12_OpinionSentiment.LABELS)
