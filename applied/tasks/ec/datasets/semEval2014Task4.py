import os
import xml.etree.ElementTree as ET
from .base import EC_Dataset, EC_DatasetItem

class __SemEval2014Task4(EC_Dataset):

    LABELS = ['positive', 'neutral', 'negative', 'conflict']

    # yield train items
    yield_train_items = lambda self: self.yield_items(
        os.path.join(self.data_base_dir, self.__class__.TRAIN_FILE))
    # yield test items
    yield_eval_items = lambda self: self.yield_items(
        os.path.join(self.data_base_dir, self.__class__.TEST_FILE))

    def yield_items(self, fpath):
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
            # get label ids
            labels = [self.__class__.LABELS.index(l) for l in labels]
            # yield item
            yield EC_DatasetItem(
                sentence=text, entity_spans=aspect_spans, labels=labels)
    
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
    """ SemEval 2014 Task 4 Laptop Dataset for Entity Classification.
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
    """ SemEval 2014 Task 4 Dataset for Entity Classification.
        Combination of the restaurant and laptop dataset.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    def yield_train_items(self) -> iter:
        # yield from restaurant and from laptop dataset
        yield from SemEval2014Task4_Restaurants.yield_train_items(self)
        yield from SemEval2014Task4_Laptops.yield_train_items(self)
    def yield_eval_items(self) -> iter:
        # yield from restaurant and from laptop dataset
        yield from SemEval2014Task4_Restaurants.yield_eval_items(self)
        yield from SemEval2014Task4_Laptops.yield_eval_items(self)

