# Entity Classification

The Entity Classification task aims to classify a given number of entities in a given text.

## Models

We currently provide the following models for this task:

- `BertForEntityClassification`

    - Basically a BERT encoder folowed by a linear classification layer. We apply a custom tokenizer which defines the special tokens `[e]` and `[/e]` to mark entities in a corpus. For classification, we pass the corpus through the BERT encoder and gather the outputs for all entity beginning markers (`[e]`). These will then be passed into the classification layer to compute the output logits.

- `KnowBertForEntityClassification`

    - Same structure as `BertForEntityClassification`
    - using KnowBERT encoder instead of standard BERT encoder

- `BertForSentencePairClassification`

    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)

- `KnowBertForSentencePairClassification`

    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)
    - using KnowBERT encoder instead of standard BERT encoder

- `BertCapsuleNetwork`
    
    - [A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1654/)

- `KnowBertCapsuleNetwork`
    
    - [A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1654/)
    - using KnowBERT encoder instead of standard BERT encoder


A custom model must have the following form:
```python
class CustomModel(EntityClassificationModel):

    # the tokenizer type to use
    # set this if the model uses some special tokenizer
    # the value will default to the standard BertTokenizer
    TOKENIZER_TYPE = transformers.BertTokenizer
    
    def __init__(self, config):
        # initialize all parameters of the model

    def prepare(self, dataset, tokenizer) -> None:
        """ Prepare the model for the dataset """
        # initialize dataset or tokenizer specific values
        # defaults to do nothing

    def build_feature_tensors(self, input_ids, entity_spans, labels, max_entities, seq_length, tokenizer) -> tuple:
        """ Build all feature tensors from a data item. """
        # This function needs to return tensors build from the provided features. 
        # Each tensor has to have the shape (n, *feature-shape), where n is the 
        # number of datapoints/items. Note that seq_length will not be set 
        # when data item for prediction is passed.
        return featureTensorA, featureTensorB, ...

    def preprocess(self, *features, tokenizer) -> tuple:
        """ Preprocess a batch of features from the prepare function. """
        # This function is called immediately before the forward call
        # and needs to return the keyword arguments for the foward call 
        # as well as the target labels for the current batch.
        return kwargs, labels

    def forward(self, **kwargs) -> tuple:
        """ The forward call of the model """
        # This function receives the keyword arguments returned by the preprocess function.
        # It needs to return the loss and entity logits of the current batch at the first positions.
        # Additional returns will be ignored.
        return loss, logits, *additionals

```

## Datasets

We currently provide the following datasets for this task:

- `SemEval2015Task12_AspectSentiment`
    - [SemEval-2015 Task 12: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S15-2082/)
    - Language: English
    - Domain: Restaurant Reviews
    - Entity Type: Aspects
    - Entity Labels:
        - positive
        - neutral
        - negative
    - [Download](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)

- `SemEval2015Task12_OpinionSentiment`
    - [SemEval-2015 Task 12: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S15-2082/)
    - Opinion Annotations by: [Coupled Multi-Layer Attentions
for Co-Extraction of Aspect and Opinion Terms](https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)
    - Language: English
    - Domain: Restaurant Reviews
    - Entity Type: Opinions
    - Entity Labels:
        - positive
        - negative
    - [Download](https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval)

- `SemEval2014Task4`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant + Laptop Reviews
    - Entity Type: Aspects
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
    - Entity Type: Aspects
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Laptops`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Laptop Reviews
    - Entity Type: Aspects
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `GermanYelpSentiment`
    - Language: German
    - Domain: Restaurant Reviews
    - Entity Type: Opinions


A custom dataset must have the following form.
```python

class CustomDataset(EntityClassificationDataset):

    # list all the possible entity labels for the dataset
    LABELS = ['YOUR', 'LABELS', ...]

    def yield_dataset_item(self, train:bool, base_data_dir:str):
        # read and process data here
        # yield tuples of the following form 
        yield (text, entity_spans, entity_labels)

```

## Evaluation

- Hyperparameters
    - Sequence Length: 128
    - Batchsize: 8
    - Learning Rate: 1e-5
    - Weight Decay: 0.01

- `SemEval2015Task12_AspectSentiment`
    
    |                 Model                |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForEntityClassification`        |    82.7    |    63.8    |   14   | bert-base-uncased            |
    | `BertForEntityClassification`        |    83.9    |    67.2    |   12   | bert-base-uncased-yelp       |
    | `BertForSentencePairClassification`  |    82.5    |    66.7    |   17   | bert-base-uncased            |
    | `BertForSentencePairClassification`  |  **86.4**  |    70.4    |   16   | bert-base-uncased-yelp       |
    | `BertCapsuleNetwork`                 |    83.4    |    65.1    |   6    | bert-base-uncased            |
    | `BertCapsuleNetwork`                 |    86.0    |  **75.3**  |   13   | bert-base-uncased-yelp       |

- `SemEval2015Task12_OpinionSentiment`
    
    |                 Model                |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForEntityClassification`        |    96.3    |    95.1    |   19   | bert-base-uncased            |
    | `BertForEntityClassification`        |    96.5    |    95.4    |   16   | bert-base-uncased-yelp       |
    | `BertForSentencePairClassification`  |    96.5    |    95.4    |   19   | bert-base-uncased            |
    | `BertForSentencePairClassification`  |    96.9    |    95.9    |   16   | bert-base-uncased-yelp       |
    | `BertCapsuleNetwork`                 |    96.9    |    95.9    |   2    | bert-base-uncased            |
    | `BertCapsuleNetwork`                 |  **97.3**  |  **96.4**  |   13   | bert-base-uncased-yelp       |

- `SemEval2014Task4`

    |                 Model                |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForEntityClassification`        |    100.0   |    100.0   |   3    | bert-base-uncased            |
    | `BertForSentencePairClassification`  |    100.0   |    100.0   |   2    | bert-base-uncased            |
    | `BertCapsuleNetwork`                 |    100.0   |    100.0   |   5    | bert-base-uncased            |

- `GermanYelpSentiment`

    |                 Model                |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForEntityClassification`        |    91.3    |    90.3    |   14   |  bert-base-german-cased      |
    | `BertForEntityClassification`        |  **94.7**  |  **94.1**  |   8    |  bert-base-german-cased-yelp |
    | `BertForSentencePairClassification`  |    92.4    |    91.5    |   7    |  bert-base-german-cased      |
    | `BertForSentencePairClassification`  |    93.7    |    93.0    |   19   |  bert-base-german-cased-yelp |
    | `BertCapsuleNetwork`                 |    91.4    |    90.3    |   4    |  bert-base-german-cased      |
    | `BertCapsuleNetwork`                 |    92.7    |    92.0    |   12   |  bert-base-german-cased-yelp |
