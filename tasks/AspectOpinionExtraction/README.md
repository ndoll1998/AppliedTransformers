# Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Models

We currently provide the following models:

- `BertForAspectOpinionExtraction`

    - Based on the BERT model for token classification. For each token, it returns aspect and opinion logits that follow the Begin-In-Out (BIO) scheme.


A custom model must have the following form:
```python
class CustomModel(AspectOpinionExtractionModel):

    # the tokenizer type to use
    # set this if the model uses some special tokenizer
    # the value will default to the standard BertTokenizer
    TOKENIZER_TYPE = transformers.BertTokenizer
    
    def __init__(self, config):
        # initialize all parameters of the model

    def prepare(self, input_ids, aspect_bio, opinion_bio, tokenizer) -> list:
        """ Prepare and extract/build all important features from a dataset item. """
        # This function needs to return tensors build from the provided features. 
        # Each tensor has to have the shape (n, *feature-shape), where n is the 
        # number of datapoints/items. Note that seq_length will not be set 
        # when data item for prediction is passed.
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
        # It needs to return the logits for aspects and opinion of the current batch 
        # separately at first two positions. Additional returns will be ignored.
        return logits, *additionals

```

## Datasets

We currently provide the following datasets for this task:

- `SemEval2015Task12`
    - [SemEval-2015 Task 12: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S15-2082/)
    - Opinion Annotations by: [Coupled Multi-Layer Attentions
for Co-Extraction of Aspect and Opinion Terms](https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)
    - Language: English
    - Domain: Restaurant Reviews
    - [Download](https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval)

- `GermanYelpDataset`
    - Language: German
    - Domain: Restaurant Reviews

A custom dataset must have the following form.
```python

class CustomDataset(AspectOpinionExtractionDataset):
    
    def yield_dataset_item(self, train:bool, base_data_dir:str):
        # read and process data here
        # yield tuples of the following form 
        yield (text, aspect_spans, opinion_spans)

```

## Evaluation

- `SemEval2015Task12`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |
    | :----------------------------------- | :---------: | :----------: | :----: |
    | `BertForAspectOpinionExtraction`     |   **81.0**  |   **79.3**   |   29   |

- `GermanYelpSentiment`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |
    | :----------------------------------- | :---------: | :----------: | :----: |
    | `BertForAspectOpinionExtraction`     |   **81.4**  |   **77.3**   |   34   |
