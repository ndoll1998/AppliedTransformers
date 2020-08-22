import os
# import pytorch and transformers
import torch
import transformers
# import base dataset
from .AspectOpinionExtractionDataset import AspectOpinionExtractionDataset


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
