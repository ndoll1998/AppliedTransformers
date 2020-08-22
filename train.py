""" Training Script for all BERT Models """

# Relation Extraction imports
# from tasks.RelationExtraction.trainer import RelationExtractionTrainer as Trainer
# from tasks.RelationExtraction.datasets import (
#     SemEval2010Task8,
#     GermanYelpRelation,
#     GermanYelpPolarity,
#     SmartdataCorpus,
# )

# Aspect Opinion Extraction imports
# from tasks.AspectOpinionExtraction.trainer import AspectOpinionExtractionTrainer as Trainer
# from tasks.AspectOpinionExtraction.datasets import (
#     SemEval2015Task12, 
#     GermanYelpDataset
# )

# Entity Classfication imports
from tasks.EntityClassification.trainer import EntityClassificationTrainer as Trainer
from tasks.EntityClassification.datasets import (
    SemEval2015Task12_AspectSentiment, 
    SemEval2015Task12_OpinionSentiment, 
    GermanYelpSentiment
)

if __name__ == '__main__':

    # create trainer
    trainer = Trainer(
        # model
        bert_base_model = 'bert-base-german-cased',
        device = 'cpu',
        # dataset
        dataset_type = GermanYelpSentiment,
        data_base_dir="./data",
        seq_length = 64,
        batch_size = 3,
        # optimizer
        learning_rate = 1e-3,
        weight_decay = 0,
    )
    exit()
    # train and save results
    trainer.train(epochs=2)
    trainer.dump('./results/')

