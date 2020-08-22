import os
# import pytorch and transformers
import torch
import transformers
# import numpy and pandas
import numpy as np
import pandas as pd
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset
# utils
from utils import match_2d_shape

class GermanYelpSentiment(EntityClassificationDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    LABELS = ["positive", "negative"]

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpSentiment.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpSentiment.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)

        # create label-to-id map
        label2id = {l:i for i, l in enumerate(GermanYelpSentiment.LABELS)}

        all_input_ids, all_entity_starts, all_labels = [], [], []
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
            opinions, sentiments = opinions['Opinion'].values, opinions['Sentiment'].values
            # remove double opinions
            opinions, unique_idx = np.unique(opinions, return_index=True)
            sentiments = sentiments[unique_idx]
            
            # no opinions found
            if len(opinions) == 0:
                continue

            # sort entities
            opinions = list(map(eval, opinions))
            sort_idx = sorted(range(len(opinions)), key=lambda i: opinions[i][0])
            opinions = [opinions[i] for i in sort_idx]
            sentiments = [sentiments[i] for i in sort_idx]
            # mark entities in sentence
            marked_sent = ''.join(
                [sent[:opinions[0][0]] + "[e]" + sent[opinions[0][0]:opinions[0][1]] + "[/e]"] + \
                [sent[o1[1]:o2[0]] + "[e]" + sent[o2[0]:o2[1]] + "[/e]" for o1, o2 in zip(opinions[:-1], opinions[1:])] + \
                [sent[opinions[-1][1]:]]
            )
            # encode sentence
            input_ids = tokenizer.encode(marked_sent)[:seq_length]
            # find all entity starts
            entity_starts = [i for i, t in enumerate(input_ids) if t == tokenizer.entity_token_id]
            # no entities in bounds
            if len(entity_starts) == 0:
                continue
            # get label ids of each entity
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
        return len(GermanYelpSentiment.LABELS)

