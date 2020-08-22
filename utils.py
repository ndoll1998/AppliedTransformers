
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

    spans = []
    begin, last_was_unk = 0, False
    for token in tokens:
        token = token.replace('##', '')
        # handle unknown
        if "[UNK]" == token:
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