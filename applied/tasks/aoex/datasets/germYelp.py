import os
import pandas as pd
from .base import AOEx_Dataset, AOEx_DatasetItem

class GermYelp(AOEx_Dataset):

    # file paths
    ANNOTATIONS_FILE = "GermanYelp/annotations.csv"
    SENTENCES_FILE = "GermanYelp/sentences.txt"

    # yield train items
    yield_train_items = lambda self: self.yield_items(train=True)
    yield_eval_items = lambda self: self.yield_items(train=False)

    def yield_items(self, train:bool) -> iter:
        # load annotations and sentences
        annotations = pd.read_csv(os.path.join(self.data_base_dir, GermYelp.ANNOTATIONS_FILE), sep="\t", index_col=0)
        with open(os.path.join(self.data_base_dir, GermYelp.SENTENCES_FILE), 'r', encoding='utf-8') as f:
            sentences = f.read().split('\n')[:-1]
        # separate all sentences into training and testing sentences
        n_train_samples = int(len(sentences) * 0.8)
        
        for k, group in annotations.groupby('SentenceID'):
            # only load train or test data, not both
            if ((k < n_train_samples) and not train) or ((k >= n_train_samples) and train):
                continue
            # read sentence, aspects and opinions
            sentence = sentences[k]
            aspects = [eval(span) for span in group['Aspect'].dropna().unique()]
            opinions = [eval(span) for span in group['Opinion'].dropna().unique()]
            # yield item
            yield AOEx_DatasetItem(
                sentence=sentence, aspect_spans=aspects, opinion_spans=opinions)