# Relation Extraction

The Relation Extraction task aims to predict the relationship type between two given entities.


## Models

We currently provide the following models:

- `BertForRelationExtraction`

    - [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158)

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

- `GermanYelp_Polarity`

    |              Model              |  Micro-F1  |  Macro-F1  | Epochs |   Pretrained Model Name      |
    | :------------------------------ | :--------: | :--------: | :----: | :--------------------------- |
    | `BertForRelationExtraction`     |    91.8    |    90.6    |   2    |  bert-base-german-cased      |
    | `BertForRelationExtraction`     |  **94.5**  |  **93.8**  |   11   |  bert-base-german-cased-yelp |

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
    | `BertForRelationExtraction`     |     100    |     100    |   11   |  bert-base-uncased           |
