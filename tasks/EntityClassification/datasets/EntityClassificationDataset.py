# import base dataset and tokenizer
from core.Dataset import BaseDataset
# import utils
from core.utils import build_token_spans

class __EntityClassificationDatasetType(type):

    @property
    def num_labels(cls) -> int:
        return len(cls.LABELS)


class EntityClassificationDataset(BaseDataset, metaclass=__EntityClassificationDatasetType):
    """ Base Dataset for the Entity Classification Task """

    # list of all labels
    LABELS = []

    def yield_item_features(self, train:bool, data_base_dir:str ='./data'):
        raise NotImplementedError()

    def build_dataset_item(self, text:str, entity_spans:list, labels:list, tokenizer):

        if labels is not None:
            # exactly one label per entity
            assert len(entity_spans) == len(labels)
            # build label-to-id map
            label2id = {l: i for i, l in enumerate(self.__class__.LABELS)}
        
        if len(entity_spans) == 0:
            return input_ids, [], []

        # tokenize text
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # sort entities by occurance in text
        sort_idx = sorted(range(len(entity_spans)), key=lambda i: entity_spans[i][0])
        entity_spans = [entity_spans[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx] if labels is not None else None
        # remove entities that overlap
        entity_spans = [entity_spans[0]] + [(b, e) for i, (b, e) in enumerate(entity_spans[1:]) if entity_spans[i][1] <= b]

        # build token spans
        token_spans = build_token_spans(tokens, text)
        # build entity token spans
        entity_token_spans, entity_id = [[-1, -1] for _ in range(len(entity_spans))], 0
        for i, (token_b, token_e) in enumerate(token_spans):
            entity_b, entity_e = entity_spans[entity_id]
            # token is part of entity
            if (entity_b <= token_b) and (token_e <= entity_e):
                if entity_token_spans[entity_id][0] == -1:
                    entity_token_spans[entity_id][0] = i
                entity_token_spans[entity_id][1] = i + 1
            # finished entity
            if token_e >= entity_e:
                entity_id += 1
                if entity_id >= len(entity_spans):
                    break

        # find all valid spans
        mask = [(b != -1) and (e != -1) for b, e in entity_token_spans]
        entity_token_spans = [span for span, valid in zip(entity_token_spans, mask) if valid]
        labels = [label2id[l] for l, valid in zip(labels, mask) if valid] if labels is not None else None
        
        # return dataset item
        return input_ids, entity_token_spans, labels