# BERT For Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Model

We use a basic BERT model for token classification. It outputs aspect and opinion logits separately and follows the Begin-In-Out (BIO) scheme to label tokens. The model is trained using the sum of both cross-entropy losses for aspects and opinions.

## Datasets

Datasets for this task must inherit the `AspectOpinionExtractionDataset` which specifies `input-ids`, `aspects` and `opinions`. The `aspects` and `opinions` fields follow an begin-in-out scheme. 

A custom dataset must have the following form.
```python

class CustomDataset(AspectOpinionExtractionDataset):
    
    def yield_dataset_item(self, train:bool, base_data_dir:str):
        # read and process data here
        # yield tuples of the following form 
        yield (text, aspect_spans, opinion_spans)

```

We currently provide the following datasets for this task:

- `SemEval2015Task12`
    - Language: English
    - Domain: Restaurant Reviews
    - [Download](https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval)

- `GermanYelpDataset`
    - Language: German
    - Domain: Restaurant Reviews

## Evaluation
