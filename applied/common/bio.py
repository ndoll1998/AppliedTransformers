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
