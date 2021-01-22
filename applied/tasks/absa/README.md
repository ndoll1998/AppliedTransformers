# Aspect-based Sentiment Analysis

The Aspect-based Sentiment Analysis (ABSA) predicts the polarity of an aspect. In contrast to the Entity Classification Task, aspects don't have do be explicitly mentioned in the provided text.

## Models

We currently provide the following models:

- `SentencePairClassifier`

    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)

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
