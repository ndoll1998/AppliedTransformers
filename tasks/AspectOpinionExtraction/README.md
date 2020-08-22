# BERT For Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Model

We use a basic BERT model for token classification. It outputs aspect and opinion logits separately and follows the Begin-In-Out (BIO) scheme to label tokens. The model is trained using the sum of both cross-entropy losses for aspects and opinions.

## Datasets

Datasets for this task must inherit the `AspectOpinionExtractionDataset` which specifies `input-ids`, `aspect-labels` and `opinion-labels` in that order.

We currently provide the following datasets for this task:

- `SemEval2015Task12`
    - Language: English
    - Domain: Restaurant Reviews
    - [Download](https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval)

- `GermanYelpDataset`
    - Language: German
    - Domain: Restaurant Reviews

## Evaluation