import os
import xml.etree.ElementTree as ET
from .base import EC_Dataset, EC_DatasetItem

class __SemEval2015Task12(EC_Dataset):
    TRAIN_FILE = None
    EVAL_FILE = None

    # yield training and evaluation items
    yield_train_items = lambda self: self.yield_items(
        os.path.join(self.data_base_dir, SemEval2015Task12_AspectPolarity.TRAIN_FILE))
    yield_eval_items = lambda self: self.yield_items(
        os.path.join(self.data_base_dir, SemEval2015Task12_AspectPolarity.EVAL_FILE))


class SemEval2015Task12_AspectPolarity(__SemEval2015Task12):
    """ Dataset for the SemEval2014 Task4 data for Aspect-based Sentiment Analysis
        Download: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
    """
    TRAIN_FILE = "SemEval2015-Task12/ABSA-15_Restaurants_Train_Final.xml"
    EVAL_FILE = "SemEval2015-Task12/ABSA15_Restaurants_Test.xml"
    LABELS = ['positive', 'neutral', 'negative']

    def yield_items(self, fpath:str) -> iter:
        # parse xml file
        tree = ET.parse(fpath)
        root = tree.getroot()
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
                # no aspects found
                if len(aspects) == 0:
                    continue
                
                # build dataset item
                yield EC_DatasetItem(
                    sentence=text,
                    entity_spans=aspects,
                    labels=[SemEval2015Task12_AspectPolarity.LABELS.index(s) for s in sentiments]
                )

class SemEval2015Task12_OpinionPolarity(__SemEval2015Task12):
    """ Dataset for the SemEval2014 Task4 data for Opinion-based Sentiment Analysis
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """
    TRAIN_FILE = "SemEval2015-Task12/sentence_res15_op"
    EVAL_FILE = "SemEval2015-Task12/sentence_restest15_op"
    LABELS = ['positive', 'negative']

    # urls map
    CAN_DOWNLOAD = True
    URL_FILE_MAP = {
        "SemEval2015-Task12/sentence_res15_op": "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_res15_op",
        "SemEval2015-Task12/sentence_restest15_op": "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_restest15_op"        
    }

    def yield_items(self, fpath:str) -> iter:
        # load file content
        with open(fpath, 'r', encoding='utf-8') as f:
            all_sents_opinions = f.read().split('\n')
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
            # build opinion spans
            opinion_pos = [sent.find(o) for o in opinions]
            opinion_spans = [(i, i + len(o)) for i, o in zip(opinion_pos, opinions)]
            # get sentiment labels
            sentiments = [(-int(i) + 1) // 2 for i in sentiments]
            # build dataset item
            yield EC_DatasetItem(
                sentence=sent, 
                entity_spans=opinion_spans, 
                labels=sentiments
            )
