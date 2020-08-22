import os
import pandas as pd
# import pytorch and transformers
import torch
import transformers
# import base dataset
from .AspectOpinionExtractionDataset import AspectOpinionExtractionDataset
# utils
from utils import build_token_spans


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
