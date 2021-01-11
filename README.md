# Applied Transformers

Applied Transformers is a project collecting state-of-the-art transformer models to tackle typical natural language processing (NLP) task. It provides PyTorch implementations of the models as well as a number of datasets for several NLP tasks.

## Tasks

The Project currently holds the following tasks:

- [AspectBasedSentimentAnalysis](applied/tasks/absa/README.md)

## How to use

Our main goal is to provide a very simple yet scalable interface for both training and infering SOTA transformer models for serveral NLP tasks. The following shows a simple example to train a `BERT` model for aspect based sentiment analysis (from `examples/train_abse.py`).

```python
import torch
from applied import encoders
from applied import optimizers
from applied.tasks import absa
import matplotlib.pyplot as plt

# create encoder
encoder = encoders.BERT.from_pretrained("bert-base-uncased")
encoder.init_tokenizer_from_pretrained("bert-base-uncased")
# create model and optimizer
model = absa.models.SentencePairClassification(encoder=encoder, 
    num_labels=absa.datasets.SemEval2014Task4.num_labels())
optim = optimizers.AdamW(model.parameters(only_head=True), lr=1e-5, weight_decay=0.01)
# create dataset and prepare it for the model
dataset = absa.datasets.SemEval2014Task4(
    data_base_dir='../data', seq_length=128, batch_size=2)
# create trainer instance and train model
trainer = absa.Trainer(model, dataset, optim)
trainer.train(epochs=10)
# save metrics and model
trainer.metrics.save("../results/ABSA-Bert/metrics.json")
torch.save(model.state_dict(), "../results/ABSA-Bert/model.bin")
# plot metrics - slice metrics to get metrics of specific epochs
fig = trainer.metrics.plot()
plt.show()
```


## TODOs
 - implement more encoders
 - make trainer iterable
 - is there any benefit of having a task instance? (core.task)
 - PyPi
