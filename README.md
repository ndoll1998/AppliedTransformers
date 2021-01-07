# Applied Transformers

Applied Transformers is a project collecting state-of-the-art transformer models to tackle typical natural language processing (NLP) task. It provides PyTorch implementations of the models as well as a number of datasets for several NLP tasks.

## Tasks

The Project currently holds the following tasks:

- `AspectBasedSentimentAnalysis`
- `AspectOpinionCoExtraction`
- `EntityClassification`
- `RelationExtraction`

## Target training example

```python
from applied import bases
from applied import optimizers
from applied.tasks import absa
import matplotlib.pyplot as plt

# create model and optimizer
base = bases.BERT.from_pretrained("bert-base-uncased")
model = absa.heads.SentencePairClassification(base=base)
optim = optimizers.AdamW(model.parameters(only_head=False), lr=1e-5, weight_decay=0.01)
# create dataset and prepare it for the model
dataset = absa.data.SemEval2014Task4(data_base_dir="./data", seq_length=128)
dataset.prepare(model)
# create a trainer instance and train the model
trainer = absa.Trainer(
    model=model,
    dataset=dataset,
    optimizer=optim
).train(
    epochs=5,
    batch_size=64,
    verbose=True
)
# save metrics and model
trainer.metrics.save("results/ABSA-Bert/")
model.save("results/ABSA-Bert/")
# plot metrics - slice metrics to get metrics of specific epochs
fig = trainer.metrics[1:5].plot()
plt.show()
```


## TODOs
 - Models need to be able to use different bases (BERT, ALBERT, KnowBERT, etc.)
 - Trainer receives model instance as argument, not model description
 - make trainer iterable
 - Ability to combine models to use the same base and apply multiple heads
   - not sure if this is a good idea because this needs to combine datasets too for combined training of multiple heads
   - is there a generic way to "synchronize" datasets?
 - move core.utils to tasks.common
 - is there any benefit of having a task instance? (core.task)
 - make the repo more package like and also PyPi
 - a dataset class should be able to yield both training and testing examples
    - `dataset.train` yields training examples
    - `dataset.eval`yields testing examples