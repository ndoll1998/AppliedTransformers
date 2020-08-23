
def match_2d_shape(L:list, shape:tuple, fill_val=0) -> list:
    """ Match the shape of an unaligned 2d nested list to the given shape
        by truncating and filling up where needed.
    """

    # get shape
    assert len(shape) == 2
    n, m = shape
    # fill up to reach shape
    L = L[:n]
    L = [l[:m] + [fill_val] * max(m - len(l), 0) for l in L]
    L = L + [[fill_val] * m] * max(len(L) - n, 0)
    # return
    return L


def build_token_spans(tokens, text):

    # not case sensitive
    tokens = [t.lower() for t in tokens]
    text = text.lower()

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