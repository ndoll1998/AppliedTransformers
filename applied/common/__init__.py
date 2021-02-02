import numpy as np
import unicodedata

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
        # remove all leading whitespaces
        begin += len(text) - len(text.lstrip())
        text = text.lstrip()
        # handle unknown
        if "[unk]" == token:
            spans.append((begin, begin))
            last_was_unk = True
            continue
        if last_was_unk:
            # find token
            n = text.find(token)
            begin, text = begin + n, text[n:]
        # make sure text starts with token
        assert text.startswith(token), "Text and tokens do not align! '%s' > '%s'" % (token, text)
        spans.append((begin, begin + len(token)))
        begin += len(token)
        text = text[len(token):]

    return spans


""" Begin-In-Out Scheme Helpers """

def build_bio_scheme(token_spans:list, entity_spans:list) -> list:

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