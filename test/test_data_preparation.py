import unittest
from applied import tasks
from applied import encoders

# create an encoder to use for testing
encoder = encoders.BERT.from_pretrained("bert-base-uncased")
encoder.init_tokenizer_from_pretrained("bert-base-uncased")

class TaskPreparationTestCase(unittest.TestCase):
    MODELS = {}
    DATASETS = []

    def test_preparation(self):
        # iterate over all assigned datasets and models
        for Dataset in self.__class__.DATASETS:
            for Model, kwargs in self.__class__.MODELS.items():
                # test preparation pipeline for each combination
                with self.subTest(Dataset=Dataset.__name__, Model=Model.__name__):
                    Dataset().prepare(Model(**kwargs))

""" Tasks """

class ABSA(TaskPreparationTestCase):
    MODELS = {
        # sentence pair classifier
        tasks.absa.models.SentencePairClassifier: {
            "encoder": encoder,
            "num_labels": 1
        },
        # capsule network
        tasks.absa.models.CapsuleNetwork: {
            "encoder": encoder,
            "num_labels": 1
        } 
    }
    DATASETS = [
        # SemEval 2014 Task 4
        tasks.absa.datasets.SemEval2014Task4_Laptops,
        tasks.absa.datasets.SemEval2014Task4_Restaurants,
        tasks.absa.datasets.SemEval2014Task4_Category,
        tasks.absa.datasets.SemEval2014Task4
    ]

class AOEX(TaskPreparationTestCase):
    MODELS = {
        tasks.aoex.models.TokenClassifier: {
            "encoder": encoder
        }
    }
    DATASETS = [
        tasks.aoex.datasets.SemEval2015Task12
    ]

class NEC(TaskPreparationTestCase):
    MODELS = {
        tasks.nec.models.SentencePairClassifier: {
            "encoder": encoder,
            "num_labels": 1
        }
    }
    DATASETS = [
        # SemEval 2014 Task 4
        tasks.nec.datasets.SemEval2014Task4_Restaurants,
        tasks.nec.datasets.SemEval2014Task4_Laptops,
        tasks.nec.datasets.SemEval2014Task4,
        # SemEval 2015 Task 12
        tasks.nec.datasets.SemEval2015Task12_AspectPolarity,
        tasks.nec.datasets.SemEval2015Task12_OpinionPolarity
    ]

class RelEx(TaskPreparationTestCase):
    MODELS = {
        tasks.relex.models.MatchingTheBlanks: {
            "encoder": encoder,
            "num_labels": 1
        }
    }
    DATASETS = [
        tasks.relex.datasets.SemEval2010Task8,
        tasks.relex.datasets.SmartdataCorpus
    ]
