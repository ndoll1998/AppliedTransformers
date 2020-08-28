# Aspect-based Sentiment Analysis

The Aspect-based Sentiment Analysis (ABSA) predicts the polarity of an aspect. In contrast to the Entity Classification Task, aspects don't have do be explicitly mentioned in the provided text.

## Models

We currently provide the following models:

- `BertForAspectBasedSentimentAnalysis`

    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)

- `BertCapsuleNetwork`
    
    - [A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1654/)


A custom model must have the following form:
```python
class CustomModel(AspectBasedSentimentAnalysisModels):

    # the tokenizer type to use
    # set this if the model uses some special tokenizer
    # the value will default to the standard BertTokenizer
    TOKENIZER_TYPE = transformers.BertTokenizer
    
    def __init__(self, config):
        # initialize all parameters of the model

    def prepare(self, input_ids, aspects_token_ids, labels, tokenizer) -> list:
        """ Prepare and extract/build all important features from a dataset item. """
        # This function needs to return tensors build from the provided features. 
        # Each tensor has to have the shape (n, *feature-shape), where n is the 
        # number of datapoints/items. Note that seq_length will not be set 
        # when data item for prediction is passed.
        return featureTensorA, featureTensorB, ...

    def preprocess(self, *features, tokenizer) -> (dict, torch.tensor):
        """ Preprocess a batch of features from the prepare function. """
        # This function is called immediately before the forward call
        # and needs to return the keyword arguments for the foward call 
        # as well as the target labels for the current batch.
        return kwargs, labels

    def forward(self, **kwargs) -> (torch.tensor, torch.tensor):
        """ The forward call of the model """
        # This function receives the keyword arguments returned by the preprocess function.
        # It needs to return the loss and polarity logits of the current batch 
        # at first two positions. Additional returns will be ignored.
        return loss, logits, *additionals

```

## Datasets

We currently provide the following datasets for this task:

- `SemEval2014Task4`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant + Laptop Reviews
    - Polarity Labels:
        - positive
        - neutral
        - negative
        - conflict
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Restaurants`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Laptops`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Laptop Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Category`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - provides only aspect-categories that are not explicitly mentioned in the text
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

A custom dataset must have the following form.
```python

class CustomDataset(AspectOpinionExtractionDataset):
    
    def yield_dataset_item(self, train:bool, base_data_dir:str):
        # read and process data here
        # yield tuples of the following form 
        yield (text, aspect_terms, labels)

```

## Evaluation
