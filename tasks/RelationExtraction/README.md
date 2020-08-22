# BERT For Entity Classification

The Relation Extraction task aims to predict the relationship type between two given entities.

## Model

We use a BERT encoder with a linear classification layer after it. We apply a custom tokenizer which defines special tokens `[e1]`, `[/e1]` and `[e2]`, `[/e2]` to mark both entities separately in a corpus. For classification, we pass the corpus through the BERT encoder and gather the outputs for the entity beginning markers (`[e1]`, `[e2]`). These will then be passed into the classification layer to compute the output logits.

## Datasets

Datasets for this task must inherit the `RelationExtractionDataset` which specifies `input-ids`, `entity-start-positions` and `relation-labels` in that order.

We currently provide the following datasets for this task:

- `SemEval2010Task8`
    - Language: English
    - Domain: General
    - Relationship Types: 
        - Component-Whole
        - Member-Collection
        - Entity-Origin
        - etc.
    - [Download](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)

- `GermanYelpRelations`
    - Language: German
    - Domain: Restaurant Reviews
    - Relationship Types:
        - True
        - False

- `GermanYelpPolarity`
    - Language: German
    - Domain: Restaurant Reviews
    - Relationship Types:
        - none
        - positive
        - negative

- `SmartdataCorpus`
    - Language: German
    - Domain: Company/Business
    - Relationship Types:
        - Acquisition
        - Insolvency
        - OrganizationLeadership
        - etc.
    - [Download](https://github.com/DFKI-NLP/smartdata-corpus/tree/master/v2_20190802)

## Evaluation