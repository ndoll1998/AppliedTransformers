# Entity Classification

The Entity Classification task aims to classify a given number of entities in a given text.

## Models

We currently provide the following models for this task:

- `BertForEntityClassification`

    - Basically a BERT encoder with a linear classification layer after it. We apply a custom tokenizer which defines the special tokens `[e]` and `[/e]` to mark entities in a corpus. For classification, we pass the corpus through the BERT encoder and gather the outputs for all entity beginning markers (`[e]`). These will then be passed into the classification layer to compute the output logits.

- `BertForSentencePairClassification`

    - "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)
    - [Paper](https://arxiv.org/abs/1903.09588)


Models for this task must inherit the `EntityClassificationModel` type. A custom model must have the following form:
```python
class CustomModel(EntityClassificationModel):

    # the tokenizer type to use
    # set this if the model uses some special tokenizer
    # the value will default to the standard BertTokenizer
    TOKENIZER_TYPE = transformers.BertTokenizer
    
    def __init__(self, config):
        # initialize all parameters of the model

    def prepare(self, input_ids, entity_spans, labels, tokenizer) -> list:
        """ Prepare and extract/build all important features from a dataset item. """
        # This function is called during the dataset creation and should handle 
        # havier computations. It needs to return a list of items extracted from
        # the provided features. One item is a tuple of aligned features.
        # The function will default to return the given item (input_ids, entity_spans, labels).
        return [itemA, itemB, itemC, ...]

    def preprocess(self, *features, tokenizer, device) -> (dict, torch.tensor):
        """ Preprocess a batch of features from the prepare function. """
        # This function is called immediately before the forward call
        # and needs to return the keyword arguments for the foward call 
        # as well as the target labels for the current batch.
        return kwargs, labels

    def forward(self, **kwargs) -> (torch.tensor, torch.tensor):
        """ The forward call of the model """
        # This function receives the keyword arguments returned by the preprocess function.
        # It needs to return the loss value and the logits of the current batch.
        # Additional returns will be ignored.
        return loss, logits, *additionals

```

## Datasets

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

