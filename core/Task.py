# import base classes
from .Predictor import BasePredictor
from .Trainer import BaseTrainer
from .Dataset import BaseDataset
from .Model import BaseModel

class Task(object):
    """ Task """

    def __init__(self, 
        predictor_type:type =None,
        trainer_type:type =None,
        base_model_type:type =None,
        base_dataset_type:type =None
    ):
        # check types
        if not issubclass(predictor_type, BasePredictor):
            raise RuntimeError("Predictor %s must extend the BasePredictor type!" % predictor_type.__name__)
        if not issubclass(trainer_type, BaseTrainer):
            raise RuntimeError("Trainer %s must extend the BaseTrainer type!" % trainer_type.__name__)
        if not issubclass(base_model_type, BaseModel):
            raise RuntimeError("Model Type %s must extend the BaseModel type!" % base_model_type.__name__)
        if not issubclass(base_dataset_type, BaseDataset):
            raise RuntimeError("Dataset Type %s must extend the BaseDataset type!" % base_dataset_type.__name__)
        # save predictor and trainer
        self.predictor_type = predictor_type
        self.trainer_type = trainer_type
        # save base model and dataset types
        self.base_model_type = base_model_type
        self.base_dataset_type = base_dataset_type

        # create model and dataset registries
        self.dataset_registry = {}
        self.model_registry = {}

    def register_dataset(self, name:str):
        """ Decorator function to register dataset types to the task """
        # check if key is already in use
        if name in self.dataset_registry:
            raise ValueError("Key %s is already in use for %s!" % (name, self.dataset_registry[name]))

        def dataset_register_decorator(dataset_type):
            # check dataset type
            if not issubclass(dataset_type, self.base_dataset_type):
                raise RuntimeError("Dataset %s must extend the %s type!" % (dataset_type.__name__, self.base_dataset_type.__name__))
            # register
            self.dataset_registry[name] = dataset_type
            # return dataset type
            return dataset_type

        # return decorator
        return dataset_register_decorator

    def register_model(self, name:str, model_type:type):
        """ Decorator function to register model types to the task """
        # check if key is already in use
        if name in self.model_registry:
            raise ValueError("Key %s is already in use for %s!" % (name, self.model_registry[name]))

        def model_register_decorator(model_type):
            # check model type
            if not issubclass(model_type, self.base_model_type):
                raise RuntimeError("Model %s must extend the %s type!" % (model_type.__name__, self.base_model_type.__name__))
            # register
            self.model_registry[name] = model_type
            # return model type
            return model_type

        # return decorator
        return model_register_decorator
