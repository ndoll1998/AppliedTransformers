# BERT For Entity Classification

The Entity Classification task aims to classify a given number of entities in a given text.

## Model

We use a BERT encoder with a linear classification layer after it. We apply a custom tokenizer which defines special tokens `[e]` and `[/e]` to mark entities in a corpus. For classification, we pass the corpus through the BERT encoder and gather the outputs for all entity beginning markers (`[e]`). These will then be passed into the classification layer to compute the output logits.

## Datasets

Datasets for this task must inherit the `EntityClassificationDataset` which specifies `input-ids`, `entity-token-spans` and `entity-labels` in that order.

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

We currently provide the following datasets for this task:

- `SemEval2015Task12_AspectSentiment`
    - Language: English
    - Domain: Restaurant Reviews
    - Entity Type: Aspects
    - [Download](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)

- `SemEval2015Task12_OpinionSentiment`
    - Language: English
    - Domain: Restaurant Reviews
    - Entity Type: Opinions
    - [Download](https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval)

- `GermanYelpSentiment`
    - Language: German
    - Domain: Restaurant Reviews
    - Entity Type: Opinions

## Evaluation