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
    """ Build token spans """

    # clean text and tokens
    # not case or accent sensitive
    tokens = [t.lower() for t in tokens]
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
        assert text.startswith(token)
        # update spans and text
        spans.append((begin, begin + len(token)))
        begin += len(token)
        text = text[len(token):]

    return spans

