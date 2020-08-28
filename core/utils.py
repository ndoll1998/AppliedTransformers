# import torch
import torch.nn as nn
# import numpy
import numpy as np
# import utils
import unicodedata
from functools import wraps


""" Tensor Helpers """

def align_shape(L:list, shape:tuple, fill_value:float =0) -> np.ndarray:
    """ Fill/Cut an arbitary deep nested list L to match the given shape. """
    assert (len(shape) > 0)

    # special case
    if (0 in shape):
        return np.full(shape, fill_value=fill_value)

    # cut off
    L = L[:shape[0]]

    if len(shape) > 1:
        # recursive match shapes
        L = np.stack([align_shape(l, shape[1:], fill_value=fill_value) for l in L], axis=0)
        # fill current dimension
        F = np.full((shape[0] - L.shape[0],) + shape[1:], fill_value=fill_value)
        L = np.concatenate((L, F), axis=0)
        # return
        return L

    # match the shape of a one dimensional array
    return np.asarray(L + [fill_value] * (shape[0] - len(L)))


""" Model Decorator Helpers """

class conditional_default_kwargs(object):
    """ Decorator to set default values for keyword arguments on condition.
        Condition can be a bool or a function returning a bool value.
        If condition is a function, then it need to accept the same arguments as f.
    """

    def __init__(self, condition:bool, **conditional_kwargs):
        # convert condition to function
        if isinstance(condition, bool):
            condition = lambda *args, **kwargs: condition

        # save condition and kwargs
        self.condition = condition
        self.conditional_kwargs = conditional_kwargs

    def __call__(self, f):

        @wraps(f)
        def wrapped_func(*args, **kwargs):
            # evaulate condition
            if self.condition(*args, **kwargs):
                kwargs = dict(
                    list(self.conditional_kwargs.items()) + list(kwargs.items())
                )

            # call function and return
            return f(*args, **kwargs)

        # return wrapped function    
        return wrapped_func

class train_default_kwargs(conditional_default_kwargs):
    """ Decorator to set the default arguments of a nn.Module 
        function when it is in training mode.
    """

    @staticmethod
    def condition(model, *args, **kwargs):
        # check model type
        if not isinstance(model, nn.Module):
            raise RuntimeError("Model is not of type pytorch.nn.Module!")
        # return condition
        return model.training

    def __init__(self, **conditional_kwargs):
        # initialize super decorator class
        conditional_default_kwargs.__init__(self, 
            condition=train_default_kwargs.condition, 
            **conditional_kwargs
        )

class eval_default_kwargs(conditional_default_kwargs):
    """ Decorator to set the default arguments of a nn.Module 
        function when it is in evaluation mode.
    """

    @staticmethod
    def condition(model, *args, **kwargs):
        # check model type
        if not isinstance(model, nn.Module):
            raise RuntimeError("Model is not of type pytorch.nn.Module!")
        # return condition
        return (not model.training)

    def __init__(self, **conditional_kwargs):
        # initialize super decorator class
        conditional_default_kwargs.__init__(self, 
            condition=eval_default_kwargs.condition, 
            **conditional_kwargs
        )


""" Word-Piece Token Helpers """

def strip_accents(text:str) -> str:
    """ Strips accents from a piece of text. """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def build_token_spans(tokens:list, text:str) -> list:

    # clean text and tokens
    # not case or accent sensitive
    tokens = [strip_accents(t.lower()) for t in tokens]
    text = strip_accents(text.lower())

    spans = []
    begin, last_was_unk = 0, False
    for token in tokens:
        token = token.replace('##', '')
        # handle unknown
        if "[unk]" == token:
            spans.append((begin, begin))
            last_was_unk = True
            continue
        if last_was_unk:
            # find token
            n = text.find(token)
            begin, text = begin + n, text[n:]
        # remove all leading whitespaces
        begin += len(text) - len(text.lstrip())
        text = text.lstrip()
        # make sure text starts with token
        assert text.startswith(token)
        spans.append((begin, begin + len(token)))
        begin += len(token)
        text = text[len(token):]

    return spans


""" Begin-In-Out Scheme Helpers """

def mark_bio_scheme(token_spans:list, entity_spans:list) -> list:

    # no entities provided
    if len(entity_spans) == 0:
        return [0] * len(token_spans)

    # sort entities by occurance in text
    entity_spans = sorted(entity_spans, key=lambda e: e[0])
    # create bio-scheme list
    bio = []
    entity_id, in_entity = 0, False
    for tb, te in token_spans:
        # get entity candidate
        eb, ee = entity_spans[entity_id]
        # check if current token is part of an entity
        if (eb <= tb) and (te <= ee):
            if in_entity:
                # already in entity
                bio.append(2)
            else:
                # new entity
                in_entity = True
                bio.append(1)
        else:
            # out of entity
            in_entity = False
            bio.append(0)

    return bio


def get_spans_from_bio_scheme(bio:list):
    
    spans, in_entity = [], False
    for i, l in enumerate(bio):

        if l == 1:
            # new entity starts
            spans.append((i, i + 1))
            in_entity = True

        elif (l == 2) and in_entity:
            # in entity
            spans[-1] = (spans[-1][0], i + 1)

        elif l == 0:
            # entity done
            in_entity = False

    return spans