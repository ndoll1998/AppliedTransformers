import json
from dataclasses import dataclass
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from typing import Tuple, Union

@dataclass
class MetricValue(object):
    value:float
    metric:type
    __str__ = lambda self: "%s: %.04f" % (self.metric.__name__, self.value)

    def table_row(self) -> str:
        n = len(self.metric.__name__)
        return " %{0}.3f ".format(n-2) % self.value

class Metric(object):
    Value:type = MetricValue
    """ Base class for Metrics """
    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError()
    def __call__(self, *args, **kwargs) -> MetricValue:
        return self.__class__.Value(
            value=self.compute(*args, **kwargs),
            metric=self.__class__
        )
    def _plot(self, track:list, axes:plt.Axes=None, **kwargs) -> Union[plt.Figure, None]:
        # create figure if neccessary
        fig, axes = (None, axes) if axes is not None else plt.subplots(1, 1, **kwargs)
        # plot
        axes.plot([v.value for v in track], label=self.__class__.__name__)
        axes.set(title=self.__class__.__name__)
        axes.legend()
        # return
        return fig

    table_header = lambda self: self.__class__.__name__

class Track(list):
    """ A Track stores the metric values of an experiment """

    def __init__(self, metric:Metric):
        assert isinstance(metric, Metric)
        self.metric = metric

    def add_entry(self,
        train_loss:float,
        eval_loss:float,
        logits:tuple,
        labels:tuple
    ) -> None:
        list.append(self, self.metric(
            train_loss=train_loss,
            eval_loss=eval_loss,
            logits=logits,
            labels=labels
        ))

    def append(self, *args, **kwargs) -> None:
        self.add_entry(*args, **kwargs)

    def plot(self, **kwargs) -> plt.Figure:
        fig = self.metric._plot(self, **kwargs)
        assert fig is not None
        return fig

    def __getitem__(self, idx) -> Union[MetricValue, "Track"]:
        # get item
        item = list.__getitem__(self, idx)
        # make track if item is a list
        if isinstance(idx, slice):
            t = Track(self.metric)
            list.__init__(t, item)
            return t
        # return single item
        return item

    def table(self) -> str:
        # create header line
        tab = " epoch | " + self.metric.table_header() + '\n'
        tab += '-' * len(tab) + '\n'
        # print all rows
        for i, metric_value in enumerate(self, 1):
            tab += " %-5i | " % i + metric_value.table_row() + "\n"
        return tab

    def save_table(self, fpath:str) -> None:
        with open(fpath, 'w+') as f:
            f.write(self.table())

""" Metric Collection """

class MetricCollectionValue(MetricValue):
    value:tuple
    __str__ = lambda self: ' - '.join(str(m) for m in self.value)
    table_row = lambda self: ' | '.join([m.table_row() for m in self.value])

class __MetricCollectionType(type):
    def __getitem__(cls, metrics:Tuple[Metric]) -> type:
        # assertions
        assert isinstance(metrics, (tuple, list))
        assert all(issubclass(m, Metric) for m in metrics)
        assert len(metrics) == len(set(metrics))
        # create metric collection type with the provided metrics
        iden = ','.join(m.__name__ for m in metrics)
        return type("MetricCollection[%s]" % iden, (MetricCollection,), {
            "_METRICS": tuple(metrics)})
    def share_axes(cls, val=True) -> type:
        cls._SHARE_AXES = val
        return cls

class MetricCollection(Metric, metaclass=__MetricCollectionType):
    """ Combine multiple Metrics into one single Metric """
    _METRICS:tuple = None
    _SHARE_AXES:bool = False
    Value = MetricCollectionValue

    def __init__(self):
        # initialize metrics
        self.metrics = tuple(M() for M in self.__class__._METRICS)
    def compute(self, *args, **kwargs) -> tuple:
        return tuple(m(*args, **kwargs) for m in self.metrics)
    def _plot(self, track:list, axes:plt.Axes=None, **kwargs) -> Union[plt.Figure, None]:
        # (not share_axes) => axes is None
        assert self.__class__._SHARE_AXES or (axes is None), \
            "Did you forget to call share_axes() on an inner MetricCollection?"
        # unpack track
        metric_tracks = zip(*(m.value for m in track))

        # create figure
        n_axes = len(self.metrics) if not self.__class__._SHARE_AXES else 1
        fig, axes = (None, axes) if axes is not None else \
            plt.subplots(n_axes, 1, sharex=True, **kwargs)
        # plot all metrics
        axes = ([axes] * len(self.metrics)) if self.__class__._SHARE_AXES else axes
        [m._plot(track=t, axes=ax) for m, t, ax in zip(self.metrics, metric_tracks, axes)]
        # set axes title
        if self.__class__._SHARE_AXES:
            title = " & ".join([m.__class__.__name__ for m in self.metrics])
            axes[0].set(title=title)
        # return figure
        return fig

    table_header = lambda self: ' | '.join([m.table_header() for m in self.metrics])

""" Loss Metrics """

TrainLoss = type("TrainLoss", (Metric,), {
    "compute": lambda self, train_loss, **kwargs: train_loss})
EvalLoss = type("EvalLoss", (Metric,), {
    "compute": lambda self, eval_loss, **kwargs: eval_loss})
Losses = MetricCollection[TrainLoss, EvalLoss].share_axes()

""" Common Metrics """

class __F1Score(Metric):
    """ Abstract F1-Score Metric """
    idx = None              # index used on label and logit tuple
    average = None          # which average to use
    ignore_label = None     # label to be ignored by the metric

    def compute(self, logits, labels, **kwargs) -> float:
        # get labels and predictions
        logits, labels = logits[self.__class__.idx], labels[self.__class__.idx]
        # discard the entries corresponding
        # to the ignore label if one is given
        if self.__class__.ignore_label is not None:
            mask = (labels != self.__class__.ignore_label)
            labels, logits = labels[mask], logits[mask, :]
        # build predictions
        predicts = logits.argmax(axis=-1)
        # compute micro-f1-score
        return f1_score(predicts, labels, average=self.__class__.average)

MicroF1Score = type("MicroF1Score", (__F1Score,), {'idx': 0, 'average': 'micro'})
MacroF1Score = type("MacroF1Score", (__F1Score,), {'idx': 0, 'average': 'macro'})
