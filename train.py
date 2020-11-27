""" Training Script for all BERT Models """

# create trainer
# set some default values
build_trainer = lambda Trainer, batch_size, Model, pretrained_name, Dataset: Trainer(
        # model and device
        model_type      = Model,
        pretrained_name = pretrained_name,
        device          = 'cuda:0',
        # dataset
        dataset_type    = Dataset,
        data_base_dir   = "./data",
        seq_length      = 128,
        batch_size      = batch_size,
        # optimizer
        learning_rate   = 1e-5,
        weight_decay    = 0.01,
    )

def AspectBasedSentimentAnalysis():
    # Aspect-based Sentiment Analysis imports
    from tasks.AspectBasedSentimentAnalysis.Trainer import AspectBasedSentimentAnalysisTrainer
    from tasks.AspectBasedSentimentAnalysis.models import BertForSentencePairClassification, BertCapsuleNetwork
    from tasks.AspectBasedSentimentAnalysis.datasets import (
        SemEval2014Task4,
        SemEval2014Task4_Laptops,
        SemEval2014Task4_Restaurants,
        SemEval2014Task4_Category,
    )

    trainer = build_trainer(
        # trainer
        Trainer         = AspectBasedSentimentAnalysisTrainer,
        batch_size      = 64,
        # model and data
        Model           = BertForSentencePairClassification,
        Dataset         = SemEval2014Task4_Restaurants,
        pretrained_name = '../bert/pretrained/bert-base-uncased-yelp',
        # pretrained_name = '../KnowBert/pretrained/bert-base-uncased-yelp-entropy',
    )
    # train and save results
    trainer.train(epochs=1)
    trainer.dump('./results/')


def AspectOpinionExtraction():

    # Aspect Opinion Extraction imports
    from tasks.AspectOpinionExtraction.Trainer import AspectOpinionExtractionTrainer
    from tasks.AspectOpinionExtraction.models import (
        BertForAspectOpinionExtraction,
        KnowBertForAspectOpinionExtraction
    )
    from tasks.AspectOpinionExtraction.datasets import (
        SemEval2015Task12, 
        GermanYelpDataset
    )

    trainer = build_trainer(
        # trainer
        Trainer         = AspectOpinionExtractionTrainer,
        batch_size      = 16,
        # model and data
        Model           = BertForAspectOpinionExtraction,
        Dataset         = GermanYelpDataset,
        # pretrained_name = 'bert-base-german-cased',
        pretrained_name = '../bert/pretrained/bert-base-german-cased-yelp',
        # pretrained_name = '../KnowBert/pretrained/bert-base-uncased-yelp-entropy',
    )
    # train and save results
    trainer.train(epochs=4)
    trainer.dump('./results/')

def EntityClassification():

    # Entity Classfication imports
    from tasks.EntityClassification.Trainer import EntityClassificationTrainer
    from tasks.EntityClassification.models import (
        BertForEntityClassification, 
        BertForSentencePairClassification,
        KnowBertForSentencePairClassification,
        BertCapsuleNetwork
    )
    from tasks.EntityClassification.datasets import (
        SemEval2015Task12_AspectPolarity, 
        SemEval2015Task12_OpinionPolarity, 
        GermanYelp_OpinionPolarity,
        GermanYelp_AspectPolarity,
        SemEval2014Task4,
        SemEval2014Task4_Laptops,
        SemEval2014Task4_Restaurants
    )

    trainer = build_trainer(
        # trainer
        Trainer         = EntityClassificationTrainer, 
        batch_size      = 16,
        # model and data
        Model           = BertForEntityClassification,
        Dataset         = GermanYelp_OpinionPolarity,
        # pretrained_name = 'bert-base-german-cased',
        pretrained_name = '../bert/pretrained/bert-base-german-cased-yelp',
        # pretrained_name = '../KnowBert/pretrained/bert-base-uncased-yelp-entropy',
    )
    # train and save results
    trainer.train(epochs=4)
    trainer.dump('./results/')

def RelationExtraction():
    
    # Relation Extraction imports
    from tasks.RelationExtraction.Trainer import RelationExtractionTrainer
    from tasks.RelationExtraction.models import (
        BertForRelationExtraction,
        KnowBertForRelationExtraction
    )
    from tasks.RelationExtraction.datasets import (
        SemEval2010Task8,
        GermanYelp_Linking,
        GermanYelp_Polarity,
        GermanYelp_LinkingAndPolarity,
        SmartdataCorpus,
    )
 
    trainer = build_trainer(
        # trainer
        Trainer         = RelationExtractionTrainer,
        batch_size      = 16,
        # model and data
        Model           = BertForRelationExtraction,
        Dataset         = GermanYelp_Linking,
        # pretrained_name = 'bert-base-german-cased',
        pretrained_name = '../bert/pretrained/bert-base-german-cased-yelp',
        # pretrained_name = '../KnowBert/pretrained/bert-base-uncased-yelp-entropy',
    )
    # train and save results
    trainer.train(epochs=4)
    trainer.dump('./results/')
    
if __name__ == '__main__':
    
    # AspectBasedSentimentAnalysis()
    AspectOpinionExtraction()
    # EntityClassification()
    # RelationExtraction()
