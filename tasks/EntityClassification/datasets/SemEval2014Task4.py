import os
# import xml parser
import xml.etree.ElementTree as ET
# import base dataset
from .EntityClassificationDataset import EntityClassificationDataset

class __SemEval2014Task4(EntityClassificationDataset):

    LABELS = ['positive', 'neutral', 'negative', 'conflict']

    # train and test files
    TRAIN_FILE = None
    TEST_FILE = None

    def yield_item_features(self, train:bool, data_base_dir:str) -> iter:
        
        # build full paths to files
        fname = SemEval2014Task4.TRAIN_FILE if train else SemEval2014Task4.TEST_FILE
        fpath = os.path.join(data_base_dir, fname)
        # parse xml file
        tree = ET.parse(fpath)
        root = tree.getroot()

        # parse all reviews
        for sentence in root:
            # get sentence
            text = sentence.find('text').text
            # get aspect terms and labels
            aspect_label_pairs = self.get_aspect_label_pairs(sentence)
            aspect_spans, labels = zip(*aspect_label_pairs) if len(aspect_label_pairs) > 0 else ([], [])
            # yield item
            yield text, aspect_spans, labels

    def get_aspect_label_pairs(self, sentence):
        raise NotImplementedError()

class SemEval2014Task4_Restaurants(__SemEval2014Task4):
    """ SemEval 2014 Task 4 Restaurant Dataset for Aspect based Sentiment Analysis.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    # files
    TRAIN_FILE = "SemEval2014-Task4/Restaurants_Train.xml"
    TEST_FILE = "SemEval2014-Task4/restaurants-trial.xml"

    def get_aspect_label_pairs(self, sentence):
        # get aspect categories and terms
        aspect_terms = sentence.find('aspectTerms')
        # load aspect label pairs
        aspect_label_pairs = []
        if aspect_terms is not None:
            aspect_label_pairs += [((int(aspect.attrib['from']), int(aspect.attrib['to'])), aspect.attrib['polarity']) for aspect in aspect_terms]
        # return
        return aspect_label_pairs


class SemEval2014Task4_Laptops(__SemEval2014Task4):
    """ SemEval 2014 Task 4 Laptop Dataset for Aspect based Sentiment Analysis.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    # files
    TRAIN_FILE = "SemEval2014-Task4/Laptops_Train.xml"
    TEST_FILE = "SemEval2014-Task4/laptops-trial.xml"

    def get_aspect_label_pairs(self, sentence):
        # get aspect categories and terms
        aspect_terms = sentence.find('aspectTerms')
        # load aspect label pairs
        aspect_label_pairs = []
        if aspect_terms is not None:
            aspect_label_pairs += [((int(aspect.attrib['from']), int(aspect.attrib['to'])), aspect.attrib['polarity']) for aspect in aspect_terms]
        # return
        return aspect_label_pairs


class SemEval2014Task4(SemEval2014Task4_Restaurants, SemEval2014Task4_Laptops):
    """ SemEval 2014 Task 4 Dataset for Aspect based Sentiment Analysis.
        Combination of the restaurant and laptop dataset.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    def yield_item_features(self, train:bool, data_base_dir:str) -> iter:
        # yield from restaurant and from laptop dataset
        yield from SemEval2014Task4_Restaurants.yield_item_features(self, train, data_base_dir)
        yield from SemEval2014Task4_Laptops.yield_item_features(self, train, data_base_dir)
