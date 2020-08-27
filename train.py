""" Training Script for all BERT Models """
import torch

# Aspect Opinion Extraction imports
from tasks.AspectOpinionExtraction.Trainer import AspectOpinionExtractionTrainer as Trainer
from tasks.AspectOpinionExtraction.models import BertForAspectOpinionExtraction
from tasks.AspectOpinionExtraction.datasets import (
    SemEval2015Task12, 
    GermanYelpDataset
)

# Aspect-based Sentiment Analysis imports
# from tasks.AspectBasedSentimentAnalysis.Trainer import AspectBasedSentimentAnalysisTrainer as Trainer
# from tasks.AspectBasedSentimentAnalysis.models import BertForSentencePairClassification
# from tasks.AspectBasedSentimentAnalysis.datasets import (
#     SemEval2014Task4
# )

# Entity Classfication imports
# from tasks.EntityClassification.Trainer import EntityClassificationTrainer as Trainer
# from tasks.EntityClassification.models import BertForEntityClassification, BertForSentencePairClassification
# from tasks.EntityClassification.datasets import (
#     SemEval2015Task12_AspectSentiment, 
#     SemEval2015Task12_OpinionSentiment, 
#     GermanYelpSentiment
# )

# Relation Extraction imports
# from tasks.RelationExtraction.Trainer import RelationExtractionTrainer as Trainer
# from tasks.RelationExtraction.models import BertForRelationExtraction
# from tasks.RelationExtraction.datasets import (
#     SemEval2010Task8,
#     GermanYelpRelation,
#     GermanYelpPolarity,
#     SmartdataCorpus,
# )

if __name__ == '__main__':

    # create trainer
    trainer = Trainer(
        # model
        model_type = BertForAspectOpinionExtraction,
        pretrained_name = 'bert-base-german-cased',
        device = 'cpu',
        # dataset
        dataset_type = SemEval2015Task12,
        data_base_dir = "./data",
        seq_length = 64,
        batch_size = 8,
        # optimizer
        learning_rate = 1e-5,
        weight_decay = 0.01,
    )
    # train and save results
    trainer.train(epochs=5)
    trainer.dump('./results/')

