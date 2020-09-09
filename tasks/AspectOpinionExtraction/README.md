# Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Models

We currently provide the following models:

- `BertForAspectOpinionExtraction`

    - Based on the BERT model for token classification. For each token, it returns aspect and opinion logits that follow the Begin-In-Out (BIO) scheme.

- `KnowBertForAspectOpinionExtraction`

    - Same structure as `BertForAspectOpinionExtraction`
    - using KnowBERT encoder instead of standard BERT encoder


A custom model must have the following form:
```python
class CustomModel(AspectOpinionExtractionModel):

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

    def build_feature_tensors(self, input_ids, aspect_bio, opinion_bio, seq_length, tokenizer) -> tuple:
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
        return kwargs, (aspect_bio_labels, opinion_bio_labels)

    def forward(self, **kwargs) -> tuple:
        """ The forward call of the model """
        # This function receives the keyword arguments returned by the preprocess function.
        # It needs to return the loss and logits for aspects and opinion of the current batch 
        # separately at first positions. Additional returns will be ignored.
        return loss, aspect_logits, opinion_logits, *additionals

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

- Hyperparameters
    - Sequence Length: 128
    - Batchsize: 8
    - Learning Rate: 1e-5
    - Weight Decay: 0.01

- `SemEval2015Task12`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |   Pretrained Model Name           |
    | :----------------------------------- | :---------: | :----------: | :----: | :-------------------------------- |
    | `BertForAspectOpinionExtraction`     |     77.9    |     77.9     |   12   |  bert-base-uncased                |
    | `BertForAspectOpinionExtraction`     |   **79.5**  |   **79.6**   |   30   |  bert-base-uncased-yelp           |
    | `KnowBertForAspectOpinionExtraction` |     78.8    |     78.3     |   28   |  bert-base-uncased-senticnet-yelp |

- `GermanYelpDataset`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |   Pretrained Model Name                |
    | :----------------------------------- | :---------: | :----------: | :----: | :------------------------------------- |
    | `BertForAspectOpinionExtraction`     |     80.2    |     76.4     |   30   |  bert-base-german-cased                |
    | `BertForAspectOpinionExtraction`     |     81.0    |     77.4     |   11   |  bert-base-german-cased-yelp           |
    | `KnowBertForAspectOpinionExtraction` |   **82.6**  |   **79.6**   |   19   |  bert-base-german-cased-senticnet-yelp |
