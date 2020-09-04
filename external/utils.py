
""" KnowBert Utilities """

from .KnowBert.src.kb.model import KnowBertHelper

def knowbert_build_caches_from_input_ids(model, input_ids, tokenizer):
    # check type of model
    if not isinstance(model, KnowBertHelper):
        raise RuntimeError("Invalid Model type! (%s)" % model.__class__.__name__)

    # get caches to restore them later
    stored_caches = model.get_kb_caches()
    stored_caches = tuple(cache for cache in stored_caches if cache is not None)
    # clear caches
    model.clear_kb_caches()
    # build knowledge base caches
    for example_input_ids in input_ids:
        # get tokens and build caches
        tokens = tokenizer.convert_ids_to_tokens(example_input_ids[example_input_ids != tokenizer.pad_token_id])
        model.stack_kb_caches(model.build_kb_caches(tokens))
    # read caches and filter out unvalid
    caches = model.get_kb_caches()
    caches = tuple(cache for cache in caches if cache is not None)
    # restore old caches
    model.set_valid_kb_caches(*stored_caches)
    # return
    return caches