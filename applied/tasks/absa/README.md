# Aspect-based Sentiment Analysis

The Aspect-based Sentiment Analysis (ABSA) predicts the polarity of an aspect. In contrast to the Entity Classification Task, aspects don't have do be explicitly mentioned in the provided text.

## Models

We currently provide the following models:

- `BertForAspectBasedSentimentAnalysis`

    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)


A custom model must have the following form:
```python
class SentencePairClassification(ABSA_Model):

    def __init__(self, encoder:Encoder):
        ABSA_Model.__init__(self, encoder=encoder)
        # initialize your model here
        # create all submodules, etc.

    def build_features_from_item(self, item:ABSA_DatasetItem) -> Tuple[FeaturePair]:
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
        labels = [f.labels for f in features]
        return (torch.LongTensor(labels),)

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

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

```

## Datasets

We currently provide the following datasets for this task:

- `SemEval2014Task4`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant + Laptop Reviews
    - Polarity Labels:
        - positive
        - neutral
        - negative
        - conflict
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Restaurants`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Laptops`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Laptop Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

- `SemEval2014Task4_Category`
    - [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://www.aclweb.org/anthology/S14-2004/)
    - Language: English
    - Domain: Restaurant Reviews
    - Polarity Labels: see `SemEval2014Task4`
    - provides only aspect-categories that are not explicitly mentioned in the text
    - [Download](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

A custom dataset must have the following form.
```python

class CustomDataset(AspectBasedSentimentAnalysisDataset):    
    def yield_train_items(self) -> Iterator[ABSA_DatasetItem]:
        # read and process training data here
        yield ABSA_DatasetItem(
            sentence=text, aspects=aspect_terms, labels=labels)
    def yield_eval_items(self) -> Iterator[ABSA_DatasetItem]:
        # read and process eval data here
        yield ABSA_DatasetItem(
            sentence=text, aspects=aspect_terms, labels=labels)

```
