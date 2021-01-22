# Aspect Opinion Co-Extraction

The Aspect Opinion Co-Extraction task aims to extraction all aspect and opinion terms in a given text.

## Models

We currently provide the following models:

- `TokenClassifier`

    - Based on the BERT model for token classification. For each token, it returns aspect and opinion logits that follow the Begin-In-Out (BIO) scheme.

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
