# Relation Extraction

The Relation Extraction task aims to predict the relationship type between two given entities.


## Models

We currently provide the following models:

- `BertForRelationExtraction`

    - [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158)


A custom model must have the following form:
```python
class CustomModel(RelationExtractionModel):

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

    def build_feature_tensors(self, input_ids, entity_span_A, entity_span_B, label, seq_length, tokenizer) -> list:
        """ Build all feature tensors from a data item. """
        # This function needs to return tensors build from the provided features. 
        # Each tensor has to have the shape (n, *feature-shape), where n is the
        # number of datapoints/items. Not that seq_length and label 
        # will not be set when data item for prediction is passed.
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
        # It needs to return the loss and relationship logits of the current batch 
        # at first two positions. Additional returns will be ignored.
        return loss, logits, *additionals

```

## Datasets

We currently provide the following datasets for this task:

- `SemEval2010Task8`
    - [SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals](https://www.aclweb.org/anthology/S10-1006/)
    - Language: English
    - Domain: General
    - Relationship Types: 
        - Component-Whole
        - Member-Collection
        - Entity-Origin
        - etc.
    - [Download](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)

- `GermanYelp_Polarity`
    - Language: German
    - Domain: Restaurant Reviews
    - Relationship Types:
        - positive
        - negative
    - Analysing the sentiment of a relation between an aspect-opinion pair

- `GermanYelp_Linking`
    - Language: German
    - Domain: Restaurant Reviews
    - Relationship Types:
        - True
        - False
    - Linking aspects and opinions to eachother

- `GermanYelp_LinkingAndPolarity`
    - Language: German
    - Domain: Restaurant Reviews
    - Relationship Types:
        - none
        - positive
        - negative
    - Linking aspects and opinions and analyse sentiment of the relation

- `SmartdataCorpus`
    - [A German Corpus for Fine-Grained Named Entity Recognition and Relation Extraction of Traffic and Industry Events](https://www.dfki.de/web/forschung/projekte-publikationen/publikationen-uebersicht/publikation/9427/)
    - Language: German
    - Domain: Company/Business
    - Relationship Types:
        - Acquisition
        - Insolvency
        - OrganizationLeadership
        - etc.
    - [Download](https://github.com/DFKI-NLP/smartdata-corpus/tree/master/v2_20190802)

A custom dataset must have the following form.
```python

class CustomDataset(RelationExtractionDataset):
    
    # list of all relation types in the dataset
    RELATIONS = ['YOUR', 'RELATIONS', ...]

    def yield_dataset_item(self, train:bool, base_data_dir:str):
        # read and process data here
        # yield tuples of the following form 
        yield (text, relation_target_A_span, relation_target_B_span, relation_type)

```

## Evaluation

- Hyperparameters
    - Sequence Length: 128
    - Batchsize: 8
    - Learning Rate: 1e-5
    - Weight Decay: 0.01

- `SemEval2010Task8`

    |              Model              |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :------------------------------ | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForRelationExtraction`     |  **83.6**  |  **79.8**  |   4    |  bert-base-uncased           |

- `GermanYelp_Linking`

    |              Model              |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :------------------------------ | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForRelationExtraction`     |  **94.8**  |  **94.7**  |   7    |  bert-base-german-cased      |
    | `BertForRelationExtraction`     |    93.7    |    93.5    |   4    |  bert-base-german-cased-yelp |

- `GermanYelp_LinkingAndPolarity`

    |              Model              |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :------------------------------ | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForRelationExtraction`     |    87.4    |    85.1    |   5    |  bert-base-german-cased      |
    | `BertForRelationExtraction`     |  **88.9**  |  **87.7**  |   7    |  bert-base-german-cased-yelp |

- `SmartdataCorpus`

    |              Model              |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :------------------------------ | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForRelationExtraction`     |   **100**  |   **100**  |   11   |  bert-base-uncased           |
