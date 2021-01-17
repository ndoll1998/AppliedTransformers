# Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Models

We currently provide the following models:

- `TokenClassifier`

    - Based on the BERT model for token classification. For each token, it returns aspect and opinion logits that follow the Begin-In-Out (BIO) scheme.


A custom model must have the following form:
```python
from models.base import AOEx_Model
from datasets.base import AOEx_DatasetItem
from applied.core.Model import FeaturePair

class Model(AOEx_Model):
    
    def __init__(self, encoder:Encoder):
        ABSA_Model.__init__(self, encoder=encoder)
        # initialize your model here
        # create all submodules, etc.

    def build_features_from_item(self, item:AOEx_DatasetItem) -> Tuple[FeaturePair]:
        """ build all feature-pairs from the provided dataset item. 
            A FeaturePair instance has to specify the text and labels. 
            Additionally you can provide the tokens.
        """
        return (
            FeaturePair(text=textA, labels=labelsA),
            FeaturePair(text=textB, labels=labelsB, tokens=tokensB),
        )
        
    def build_target_tensors(self, features:Tuple[FeaturePair], seq_length:int) -> Tuple[torch.LongTensor]:
        """ build all target tensors from the given features. 
            Note that even in the case of only one target tensor, 
            you still have to return a tuple of tensors. 
        """
        return labels

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


    def forward(self, 
        input_ids, 
        attention_mask=None, 
        token_type_ids=None
    ):
        """ the forward pass for the given model """
        # pass through base model
        last_hidden_state, pooled_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[:2]
        # apply model
        # ...

        # return logits
        return logits

    def loss(self, *logits, *labels):
        """ compute loss from logits and labels """
        return loss
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

- `GermYelp`
    - Language: German
    - Domain: Restaurant Reviews

A custom dataset must have the following form.
```python
from typing import Iterator
from datasets.base import AOEx_Dataset, AOEx_DatasetItem

class CustomDataset(AOEx_Dataset):
    def yield_train_items(self) -> Iterator[AOEx_DatasetItem]:
        # read and process training data here
        yield AOEx_DatasetItem(
            sentence=text, aspect_spans=aspect_spans, opinion_spans=opinion_spans)
    def yield_eval_items(self) -> Iterator[AOEx_DatasetItem]:
        # read and process evaluation data here
        yield AOEx_DatasetItem(
            sentence=text, aspect_spans=aspect_spans, opinion_spans=opinion_spans)

```

## Evaluation

- Hyperparameters
    - Sequence Length: 128
    - Batchsize: 8
    - Learning Rate: 1e-5
    - Weight Decay: 0.01

- `SemEval2015Task12`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :---------: | :----------: | :----: | :--------------------------- |
    | `BertForAspectOpinionExtraction`     |     77.9    |     77.9     |   12   |  bert-base-uncased           |
    | `BertForAspectOpinionExtraction`     |   **79.8**  |   **79.0**   |   30   |  bert-base-uncased-yelp      |

- `GermanYelpDataset`

    |                 Model                |  Aspect-F1  |  Opinion-F1  | Epochs |   Pretrained Model Name      |
    | :----------------------------------- | :---------: | :----------: | :----: | :--------------------------- |
    | `BertForAspectOpinionExtraction`     |     80.2    |   **76.4**   |   30   |  bert-base-german-cased      |
    | `BertForAspectOpinionExtraction`     |   **81.2**  |   **76.4**   |   29   |  bert-base-german-cased-yelp |
