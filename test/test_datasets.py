import unittest
import types
from applied import tasks
from applied import encoders

""" Preparation """

# create encoder
encoder = encoders.BERT.from_pretrained("bert-base-uncased")
encoder.init_tokenizer_from_pretrained("bert-base-uncased")

class __DatasetTestCaseType(type):
    def __new__(cls, name, bases, attrs):
        Test = type.__new__(cls, name, bases, attrs)
        for Dataset in Test.DATASETS:
            fn = lambda self: Dataset().prepare(Test.MODEL)
            setattr(Test, "test_%s" % Dataset.__name__, fn)
        return Test

class DatasetTestCase(unittest.TestCase, metaclass=__DatasetTestCaseType):
    MODEL = None
    DATASETS = []


""" Datasets """

class ABSA(DatasetTestCase):
    MODEL = tasks.absa.models.SentencePairClassifier(
        encoder=encoder, 
        num_labels=1
    )
    DATASETS = [
        tasks.absa.datasets.SemEval2014Task4_Laptops,
        tasks.absa.datasets.SemEval2014Task4_Restaurants,
        tasks.absa.datasets.SemEval2014Task4_Category,
        tasks.absa.datasets.SemEval2014Task4
    ]

class AOEX(DatasetTestCase):
    MODEL = tasks.aoex.models.TokenClassifier(encoder=encoder)
    DATASETS = [
        tasks.aoex.datasets.SemEval2015Task12
    ]

class NEC(DatasetTestCase):
    MODEL = tasks.nec.models.SentencePairClassifier(
        encoder=encoder,
        num_labels=1
    )
    DATASETS = [
        # SemEval 2014 Task 4
        tasks.nec.datasets.SemEval2014Task4_Restaurants,
        tasks.nec.datasets.SemEval2014Task4_Laptops,
        tasks.nec.datasets.SemEval2014Task4,
        # SemEval 2015 Task 12
        tasks.nec.datasets.SemEval2015Task12_AspectPolarity,
        tasks.nec.datasets.SemEval2015Task12_OpinionPolarity
    ]

class RelEx(DatasetTestCase):
    MODEL = tasks.relex.models.MatchingTheBlanks(
        encoder=encoder,
        num_labels=1
    )
    DATASETS = [
        tasks.relex.datasets.SemEval2010Task8
    ]
