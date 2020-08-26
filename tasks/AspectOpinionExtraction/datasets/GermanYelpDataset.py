import os
import pandas as pd
# import base dataset
from .AspectOpinionExtractionDataset import AspectOpinionExtractionDataset


class GermanYelpDataset(AspectOpinionExtractionDataset):

    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(data_base_dir, GermanYelpDataset.ANNOTATIONS_FILE), sep="\t", index_col=0)
        sentences = open(os.path.join(data_base_dir, GermanYelpDataset.SENTENCES_FILE), 'r', encoding='utf-8').read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)
        
        for k, group in annotations.groupby('SentenceID'):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # read sentence, aspects and opinions
            sentence = sentences[k]
            aspects = [eval(span) for span in group['Aspect'] if span == span]
            opinions = [eval(span) for span in group['Opinion'] if span == span]
            # yield item features
            yield sentence, aspects, opinions
