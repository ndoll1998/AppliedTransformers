import os
# import xml parser
import xml.etree.ElementTree as ET
# import base dataset
from .AspectBasedSentimentAnalysisDataset import AspectBasedSentimentAnalysisDataset

class SemEval2014Task4(AspectBasedSentimentAnalysisDataset):
    """ SemEval 2014 Task 4 Dataset for Aspect based Sentiment Analysis 
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """

    LABELS = ['positive', 'neutral', 'negative', 'conflict']

    # files
    TRAIN_FILE = "SemEval2014-Task4/Restaurants_Train.xml"
    TEST_FILE = "SemEval2014-Task4/restaurants-trial.xml"

    def yield_item_features(self, train:bool, data_base_dir:str) -> list:
        
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
            aspect_label_pairs = [(aspect.attrib['category'], aspect.attrib['polarity']) for aspect in sentence.find('aspectCategories')]
            aspect_terms, labels = zip(*aspect_label_pairs)
            # yield item
            yield text, aspect_terms, labels