import os
import json
import pandas as pd
# import pytorch and transformers
import torch
import transformers
# import base dataset
from base import BaseDataset
# utils
from utils import build_token_spans


class AspectOpinionExtractionDataset(BaseDataset):
    """ Base Dataset for the Aspect-Opinion Extraction Task """

    def __init__(self, input_ids, labels_a, labels_o):
        # initialize dataset
        torch.utils.data.TensorDataset.__init__(self, input_ids, labels_a, labels_o)


""" SemEval2015 Task12 """

def mark_bio(sequence, subsequences):
    # mark all subsequences in sequence in bio-scheme
    marked = [0] * len(sequence)
    for subseq in subsequences:
        for i in range(len(sequence) - len(subseq)):
            if sequence[i:i+len(subseq)] == subseq:
                # mark b => 1 and i => 2
                marked[i] = 1
                marked[i+1:i+len(subseq)] = [2] * (len(subseq)-1)
    return marked

class SemEval2015Task12(AspectOpinionExtractionDataset):
    """ Dataset for the SemEval2014 Task4 data for Aspect-Opinion Extraction
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """

    TRAIN_ASPECT_TERMS_FILE = "SemEval2015-Task12/aspectTerm_res15.txt"
    TRAIN_SENTENCE_AND_OPINIONS_FILE = "SemEval2015-Task12/sentence_res15_op.txt"

    TEST_ASPECT_TERMS_FILE = "SemEval2015-Task12/aspectTerm_restest15.txt"
    TEST_SENTENCE_AND_OPINIONS_FILE = "SemEval2015-Task12/sentence_restest15_op.txt"
    
    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # get files for training or evaluation
        aspect_fname = SemEval2015Task12.TRAIN_ASPECT_TERMS_FILE if train else SemEval2015Task12.TEST_ASPECT_TERMS_FILE
        sent_opinion_fname = SemEval2015Task12.TRAIN_SENTENCE_AND_OPINIONS_FILE if train else SemEval2015Task12.TEST_SENTENCE_AND_OPINIONS_FILE
        # build full paths to files
        aspect_fpath = os.path.join(data_base_dir, aspect_fname)
        sent_opinion_fpath = os.path.join(data_base_dir, sent_opinion_fname)

        # load file contents
        with open(aspect_fpath, 'r', encoding='utf-8') as f:
            all_aspects = f.read().replace('NULL', '').split('\n')
        with open(sent_opinion_fpath, 'r', encoding='utf-8') as f:
            all_sents_opinions = f.read().split('\n')
        assert len(all_aspects) == len(all_sents_opinions)

        all_input_ids, all_labels_a, all_labels_o = [], [], []
        # preprocess data
        for sent_opinions, aspects in zip(all_sents_opinions, all_aspects):
            # separate sentence from opinions
            sent, opinions = sent_opinions.split('##') if '##' in sent_opinions else (sent_opinions, '')
            # get aspects and opinions
            opinions = [o.strip()[:-3] for o in opinions.split(',')] if len(opinions) > 0 else []
            aspects = [a.strip() for a in aspects.split(',')] if len(aspects) > 0 else []
            # tokenize text and all aspect and opinion
            tokens = tokenizer.tokenize(sent)[:seq_length]
            aspect_tokens = map(tokenizer.tokenize, aspects)
            opinion_tokens = map(tokenizer.tokenize, opinions)
            # create labels and convert tokens to ids
            labels_a = mark_bio(tokens, aspect_tokens)
            labels_o = mark_bio(tokens, opinion_tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # pad to fit sequence length
            input_ids += [tokenizer.pad_token_id] * (seq_length - len(input_ids))
            labels_a += [-1] * (seq_length - len(labels_a))
            labels_o += [-1] * (seq_length - len(labels_o))
            # add to lists
            all_input_ids.append(input_ids)
            all_labels_a.append(labels_a)
            all_labels_o.append(labels_o)

        # create tensors
        input_ids = torch.LongTensor(all_input_ids)
        labels_a = torch.LongTensor(all_labels_a)
        labels_o = torch.LongTensor(all_labels_o)

        # initialize dataset
        AspectOpinionExtractionDataset.__init__(self, input_ids, labels_a, labels_o)


""" German Yelp """

class GermanYelpDataset(AspectOpinionExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpDataset.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpDataset.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)
        
        all_input_ids, all_aspects_bio, all_opinions_bio = [], [], []
        for k, group in annotations.groupby('SentenceID'):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # read sentence, aspects and opinions
            sentence = sentences[k].lower() if tokenizer.basic_tokenizer.do_lower_case else sentences[k]
            aspects = [eval(span) for span in group['Aspect'] if span == span]
            opinions = [eval(span) for span in group['Opinion'] if span == span]
            # tokenize sentence and build token spans
            tokens = tokenizer.tokenize(sentence)[:seq_length]
            spans = build_token_spans(tokens, sentence)
            # build aspect and opinion labels
            aspect_mask = [([(ba <= bs) and (es <= ea) for ba, ea in aspects] + [True]).index(True) for (bs, es) in spans]
            opinion_mask = [([(bo <= bs) and (es <= eo) for bo, eo in opinions] + [True]).index(True) for (bs, es) in spans]
            # build begin-in-out scheme
            aspects_bio = [0 if len(aspects) == k else 1 if (i == 0) or (aspect_mask[i-1] != k) else 2 for i, k in enumerate(aspect_mask)]
            opinions_bio = [0 if len(opinions) == k else 1 if (i == 0) or (opinion_mask[i-1] != k) else 2 for i, k in enumerate(opinion_mask)]
            # convert tokens to ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # pad to fit sequence length
            input_ids += [tokenizer.pad_token_id] * (seq_length - len(input_ids))
            aspects_bio += [-1] * (seq_length - len(aspects_bio))
            opinions_bio += [-1] * (seq_length - len(opinions_bio))
            # add to list
            all_input_ids.append(input_ids)
            all_aspects_bio.append(aspects_bio)
            all_opinions_bio.append(opinions_bio)

        # create tensors
        input_ids = torch.LongTensor(all_input_ids)
        labels_a = torch.LongTensor(all_aspects_bio)
        labels_o = torch.LongTensor(all_opinions_bio)

        # initialize dataset
        AspectOpinionExtractionDataset.__init__(self, input_ids, labels_a, labels_o)
