""" Training Script for all BERT Models """


# Aspect Opinion Extraction imports
from tasks.AspectOpinionExtraction.trainer import AspectOpinionExtractionTrainer as Trainer
from tasks.AspectOpinionExtraction.models import BertForAspectOpinionExtraction
from tasks.AspectOpinionExtraction.datasets import (
    SemEval2015Task12, 
    GermanYelpDataset
)

# Entity Classfication imports
# from tasks.EntityClassification.trainer import EntityClassificationTrainer as Trainer
# from tasks.EntityClassification.models import BertForEntityClassification
# from tasks.EntityClassification.datasets import (
#     SemEval2015Task12_AspectSentiment, 
#     SemEval2015Task12_OpinionSentiment, 
#     GermanYelpSentiment
# )

# Relation Extraction imports
# from tasks.RelationExtraction.trainer import RelationExtractionTrainer as Trainer
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
        dataset_type = GermanYelpDataset,
        data_base_dir = "./data",
        seq_length = 64,
        batch_size = 3,
        # optimizer
        learning_rate = 1e-3,
        weight_decay = 0,
    )
    # train and save results
    trainer.train(epochs=2)
    trainer.dump('./results/')

