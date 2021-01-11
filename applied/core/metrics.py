import json
from dataclasses import dataclass
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

""" Basic Loss Metrics """

@dataclass
class LossMetricsEntry(object):
    train_loss:float
    eval_loss:float
    @staticmethod
    def create(train_loss, eval_loss, logits=None, labels=None):
        return LossMetricsEntry(train_loss=train_loss, eval_loss=eval_loss)
    def __str__(self) -> str:
        return "train-loss: %.04f - eval-loss: %.04f" % (self.train_loss, self.eval_loss)

class LossMetrics(list):
    # itemtype
    ENTRY_TYPE = LossMetricsEntry

    def new_entry(self, *args, **kwargs):
        self.append(self.__class__.ENTRY_TYPE.create(*args, **kwargs))

    @property
    def unpacked_metrics(self) -> dict:
        if len(self) == 0: return {}
        return {name: [getattr(e, name) for e in self] for name in self[0].__dataclass_fields__.keys()}

    def plot(self, figsize=(8, 5)) -> plt.Figure:
        # create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plot all metrics in one axes
        for name, series in self.unpacked_metrics.items():
            ax.plot(series, label=name)
        ax.legend()
        ax.set(xlabel="Epochs", ylabel="Metrics")
        # return figure
        return fig

    def save(self, fpath:str) -> None:
        metrics_serialized = [{name: getattr(e, name) for name in e.__dataclass_fields__.keys()} for e in self]
        with open(fpath, 'w+') as f:
            f.write(json.dumps(metrics_serialized))

    @classmethod
    def load(cls, fpath:str) -> "Metrics":
        pass

    def __getitem__(self, idx):
        item = list.__getitem__(self, idx)
        return item if not isinstance(item, list) else self.__class__(item)


""" Single Target Metrics """

@dataclass
class SingleTargetMetricsEntry(LossMetricsEntry):
    micro_f1:float
    macro_f1:float

    @staticmethod
    def create(train_loss, eval_loss, logits, labels):
        # unpack logits and labels and get predictions
        logits, labels = logits[0], labels[0].numpy()
        predicts = logits.max(dim=-1)[1].numpy()
        # compute f1-scores
        micro_f1 = f1_score(predicts, labels, average='micro')
        macro_f1 = f1_score(predicts, labels, average='macro')
        # create instance
        return SingleTargetMetricsEntry(
            train_loss=train_loss,
            eval_loss=eval_loss,
            micro_f1=micro_f1,
            macro_f1=macro_f1
        )

    def __str__(self) -> str:
        return "train-loss: %.04f - eval-loss: %.04f - micro-f1: %.04f - macro-f1: %.04f" % (
            self.train_loss, self.eval_loss, self.micro_f1, self.macro_f1)

class SingleTargetMetrics(LossMetrics):
    ENTRY_TYPE = SingleTargetMetricsEntry