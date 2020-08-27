
def RelationExtraction():

    # Relation Extraction imports
    from tasks.RelationExtraction.Predictor import RelationExtractionPredictor
    from tasks.RelationExtraction.models import BertForRelationExtraction
    from tasks.RelationExtraction.datasets import (
        SemEval2010Task8,
        GermanYelpRelation,
        GermanYelpPolarity,
        SmartdataCorpus,
    )

    # create predictor
    predictor = RelationExtractionPredictor(
        model_type = BertForRelationExtraction,
        pretrained_name = 'bert-base-uncased',
        device = 'cpu',
        # dataset
        dataset_type=SemEval2010Task8
    )

    # predict label
    label = predictor(
        text="Nasa sends names into space.",
        entity_span_A=(11, 16),
        entity_span_B=(22, 27)
    )

    # print
    print("Label: %s" % label)


def EntityClassification():

    # Entity Classfication imports
    from tasks.EntityClassification.Predictor import EntityClassificationPredictor
    from tasks.EntityClassification.models import BertForEntityClassification, BertForSentencePairClassification
    from tasks.EntityClassification.datasets import (
        SemEval2015Task12_AspectSentiment, 
        SemEval2015Task12_OpinionSentiment, 
        GermanYelpSentiment
    )

    # create predictor
    predictor = EntityClassificationPredictor(
        model_type = BertForSentencePairClassification,
        pretrained_name = 'bert-base-uncased',
        device = 'cpu',
        # dataset
        dataset_type=SemEval2015Task12_AspectSentiment
    )

    # predict entity labels
    label = predictor(
        text="Nasa sends names into space.",
        entity_spans=[(11, 16), (22, 27)]
    )

    # print
    print("Labels:", label)


def AspectOpinionExtraction():

    # Aspect Opinion Extraction imports
    from tasks.AspectOpinionExtraction.Predictor import AspectOpinionExtractionPredictor
    from tasks.AspectOpinionExtraction.models import BertForAspectOpinionExtraction
    from tasks.AspectOpinionExtraction.datasets import (
        SemEval2015Task12, 
        GermanYelpDataset
    )

    # create predictor
    predictor = AspectOpinionExtractionPredictor(
        model_type = BertForAspectOpinionExtraction,
        pretrained_name = 'bert-base-uncased',
        device = 'cpu',
        # dataset
        dataset_type=SemEval2015Task12
    )

    # predict aspect and opinions
    aspects, opinions = predictor(text="The coffee was hot and tasty.")

    # print
    print("Aspects:", aspects)
    print("Opinions:", opinions)
    

def AspectBasedSentimentAnalysis():

    # Aspect Opinion Extraction imports
    from tasks.AspectBasedSentimentAnalysis.Predictor import AspectBasedSentimentAnalysisPredictor
    from tasks.AspectBasedSentimentAnalysis.models import BertForSentencePairClassification
    from tasks.AspectBasedSentimentAnalysis.datasets import (
        SemEval2014Task4
    )

    # create predictor
    predictor = AspectBasedSentimentAnalysisPredictor(
        model_type = BertForSentencePairClassification,
        pretrained_name = 'bert-base-uncased',
        device = 'cpu',
        # dataset
        dataset_type=SemEval2014Task4
    )

    # predict label
    polarity = predictor(
        text="The waiter forgot what we ordered.",
        aspect_terms=["service"]
    )

    # print
    print("Polarity:", polarity)

if __name__ == '__main__':

    # RelationExtraction()
    # EntityClassification()
    # AspectOpinionExtraction()
    AspectBasedSentimentAnalysis()