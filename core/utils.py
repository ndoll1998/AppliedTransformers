import unicodedata

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
        print(text[:30], '<-', token)
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