# Applied Transformers

Applied Transformers is a project collecting state-of-the-art transformer models to tackle typical natural language processing (NLP) task. It provides PyTorch implementations of the models as well as a number of datasets for several NLP tasks. The beauty of Applied Transformers is that one can easily switch between different encoders (i.e. BERT, ALBERT, KnowBERT, etc.) indipendent of the task or classifier.

## Tasks

The Project currently holds the following tasks:

- [AspectBasedSentimentAnalysis](applied/tasks/absa/README.md)
- [AspectOpinionExtraction](applied/tasks/aoex/README.md)
- [EntityClassification](applied/tasks/ec/README.md)

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
model = absa.models.SentencePairClassifier(encoder=encoder, 
    num_labels=absa.datasets.SemEval2014Task4.num_labels())
optim = optimizers.AdamW(model.parameters(only_head=True), lr=1e-5, weight_decay=0.01)
# create dataset and prepare it for the model
dataset = absa.datasets.SemEval2014Task4(
    data_base_dir='../data', seq_length=128, batch_size=2)
# create trainer instance and train model
trainer = absa.Trainer(
    model=model, 
    dataset=dataset,
    optimizer=optim
).train(epochs=2)
# save metrics and model
trainer.metrics.save_table("../results/ABSA-Bert/metrics.table")
torch.save(model.state_dict(), "../results/ABSA-Bert/model.bin")
# plot metrics
fig = trainer.metrics.plot()
plt.show()
```

## Development
The overall architecture of this project aims to minimize the effort needed to implemente new ideas. This includes new models, datasets and also whole new tasks. The following section will describe the environment of a single task.

### Task Directory

In general a task is a directory with the following structure
```
+-- taskA
|   +-- models
|   |   +-- __init__.py
|   |   +-- base.py
|   |   +-- ...
|   +-- datasets
|   |   +-- __init__.py
|   |   +-- base.py
|   |   +-- ...
|   +-- __init__.py
|   +-- trainer.py
```
We'll give a quick overview on the initial files and their purpose
 - `models/base.py`
    ```python
    """ 
    Create the base model type. Usually there is nothing really to define here. 
    The base class is just for typechecking. 
    """
    from applied.core.model import Model
    class BaseModel(Model): pass
    ```
 - `datasets/base.py`
    ```python
    """ Create the base dataset type and the dataset item type. """
    from applied.core.dataset import Dataset, DatasetItem
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class MyDatasetItem(DatasetItem):
        """ Define all the features that one single dataset item should contain. """
        text:str,
        labels:tuple
        ...
    class BaseDataset(Dataset): pass
        """ Again this class is just for typechecking. So usually nothing really to do here. """

    ```
- `trainer.py`
    ```python
    """ 
    This file defines the trainer of the task. In most cases the trainer 
    only differs from the standard trainer by the metrics that are to be tracked. 
    """
    from applied.core.trainer import Trainer as BaseTrainer
    from applied.core.metrics import MetricCollection, Losses, MicroF1Score, MacroF1Score
    # import model and dataset
    from .models.base import BaseModel
    from .datasets.base import BaseDataset

    class Trainer(BaseTrainer):
        # model and dataset type
        BASE_MODEL_TYPE = BaseModel
        BASE_DATASET_TYPE = BaseDataset
        # metrics type
        # change this as needed
        METRIC_TYPE = MetricCollection[Losses, ...]
    ```

### Custom Models and Datasets
With the task directory set up one can add new models. Just add a file to the models folder and implement the custom model. Typically the model class is of the following form:

```python
from .base import BaseModel
from ..datasets.base import MyDatasetItem
from applied.core.Model import Encoder, InputFeatures

@dataclass
class CustomInputFeatures(InputFeatures):
    """ specify additional input features for the custom model
        The features should be of a torch tensor type
        They will be passed to the forward function in the 
        same order as they are defined here
    """
    # Note that you do not need to set this to a tensor by hand 
    # but the values will be collected and converted to the 
    # specified tensor type in preprocessing
    additional_input:torch.Tensor
    ...

class CustomModel(BaseModel):

    def __init__(self, encoder:Encoder):
        BaseModel.__init__(self, encoder=encoder)
        # initialize your model here
        # create all submodules, etc.

    def build_features_from_item(self, item:MyDatasetItem) -> Tuple[FeaturePair]:
        """ build all feature-pairs from the provided dataset item. 
            A FeaturePair instance has to specify the text and labels. 
            Additionally you can provide the tokens. 
            Note that you can provide labels in any form. So this is NOT a 
            restriction to only sentence-level or token-level tasks.
        """
        return (
            CustomInputFeatures(text=textA, labels=labelsA),
            CustomInputFeatures(text=textB, labels=labelsB, tokens=tokensB),
        )

    def build_target_tensors(self, features:Tuple[CustomInputFeatures]) -> Tuple[torch.LongTensor]:
        """ build all target tensors from the given features. 
            Note that even in the case of only one target tensor, 
            you still have to return a tuple of tensors.
            Also make sure that the first dimension of each label tensor 
            corresponds to the examples, i.e. it should have the size of len(features).
        """
        labels = [f.labels for f in features]
        return (torch.LongTensor(labels),)

    def forward(self, 
        # encoder input arguments
        input_ids, 
        attention_mask=None, 
        token_type_ids=None,
        # additional input arguments
        additional_input=None,
        ...
    ):
        """ the forward pass for the given model """
        # pass through encoder
        last_hidden_state, pooled_output = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[:2]
        # apply model
        # ...

        # return logits
        return logits

    def loss(self, logitsA, logitsB, labelsA, labelsB):
        """ Compute loss """
        return F.cross_entropy(logitsA, labelsA)
```

Implementing a custom dataset is just as easy:

```python
from .base import BaseDataset, MyDatasetItem

class CustomDataset(Dataset):
    def yield_train_items(self) -> iter:
        # load all training data and yield the dataset items one by one
        base_data_dir = self.base_data_dir
        yield MyDatasetItem(...)
    def yield_eval_items(self) -> iter:
        # load all evaluation data and yield the dataset items one by one
        base_data_dir = self.base_data_dir
        yield MyDatasetItem(...)
```
## TODOs
 - implement more encoders
 - make trainer iterable
 - is there any benefit of having a task instance? (core.task)
 - PyPi
