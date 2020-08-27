import os
# import xml parser
import xml.etree.ElementTree as ET
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset


class SemEval2015Task12_AspectSentiment(EntityClassificationDataset):
    """ Dataset for the SemEval2014 Task4 data for Aspect-based Sentiment Analysis
        Download: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
    """

    TRAIN_FILE = "SemEval2015-Task12/ABSA-15_Restaurants_Train_Final.xml"
    TEST_FILE = "SemEval2015-Task12/ABSA15_Restaurants_Test.xml"

    LABELS = ['positive', 'neutral', 'negative']

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # build full paths to files
        fname = SemEval2015Task12_AspectSentiment.TRAIN_FILE if train else SemEval2015Task12_AspectSentiment.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

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
                yield text, aspects, sentiments



class SemEval2015Task12_OpinionSentiment(EntityClassificationDataset):
    """ Dataset for the SemEval2014 Task4 data for Opinion-based Sentiment Analysis
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """

    TRAIN_FILE = "SemEval2015-Task12/sentence_res15_op"
    TEST_FILE = "SemEval2015-Task12/sentence_restest15_op"

    LABELS = ['positive', 'negative']

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):

        # build full paths to files
        fname = SemEval2015Task12_OpinionSentiment.TRAIN_FILE if train else SemEval2015Task12_OpinionSentiment.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)

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
            sentiments = [SemEval2015Task12_OpinionSentiment.LABELS[(-int(i) + 1) // 2] for i in sentiments]
            
            yield sent, opinion_spans, sentiments
            
