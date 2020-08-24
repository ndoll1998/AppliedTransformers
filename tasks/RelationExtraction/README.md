# Relation Extraction

The Relation Extraction task aims to predict the relationship type between two given entities.

## Model

We use a BERT encoder with a linear classification layer after it. We apply a custom tokenizer which defines special tokens `[e1]`, `[/e1]` and `[e2]`, `[/e2]` to mark both entities separately in a corpus. For classification, we pass the corpus through the BERT encoder and gather the outputs for the entity beginning markers (`[e1]`, `[e2]`). These will then be passed into the classification layer to compute the output logits.

## Datasets

Datasets for this task must inherit the `RelationExtractionDataset` which specifies `input-ids`, `relation-target-A-spans`, `relation-target-B-spans` and `relation-labels` in that order.

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
