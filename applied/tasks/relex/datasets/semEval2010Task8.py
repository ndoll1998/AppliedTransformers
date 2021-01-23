import os, re
from .base import RelExDataset, RelExDatasetItem

class SemEval2010Task8(RelExDataset):
    """ SemEval2010 Task8 Dataset
        Download: https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip
    """

    TRAIN_FILE = "SemEval2010-Task8/SemEval2010_task8_training/TRAIN_FILE.TXT"
    EVAL_FILE = "SemEval2010-Task8/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    LABELS = [
        "Other",
        "Component-Whole(e2,e1)",       "Component-Whole(e1,e2)",
        "Instrument-Agency(e2,e1)",     "Instrument-Agency(e1,e2)",
        "Member-Collection(e2,e1)",     "Member-Collection(e1,e2)",
        "Cause-Effect(e2,e1)",          "Cause-Effect(e1,e2)",
        "Entity-Destination(e2,e1)",    "Entity-Destination(e1,e2)",
        "Content-Container(e2,e1)",     "Content-Container(e1,e2)",
        "Message-Topic(e2,e1)",         "Message-Topic(e1,e2)",
        "Product-Producer(e2,e1)",      "Product-Producer(e1,e2)",
        "Entity-Origin(e2,e1)",         "Entity-Origin(e1,e2)"
    ]

    yield_train_items = lambda self: self.yield_item(
        os.path.join(self.data_base_dir, SemEval2010Task8.TRAIN_FILE))
    yield_eval_items = lambda self: self.yield_item(
        os.path.join(self.data_base_dir, SemEval2010Task8.EVAL_FILE))

    def yield_item(self, fpath:str) -> iter:
                
        # load data
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        # read examples
        for sent_line, relation_line in zip(lines[::4], lines[1::4]):
            # get text
            sent = sent_line.split('\t')[1].strip()
            # clean up sentence
            assert sent[0] == sent[-1] == '"'
            sent = sent[1:-1]
            # find entities in sentence
            entity_A = re.search(r'<e1>(.*)</e1>', sent)
            entity_B = re.search(r'<e2>(.*)</e2>', sent)
            # get spans from matches with markers
            entity_span_A = (entity_A.start(), entity_A.end() - 4 - 5)
            entity_span_B = (entity_B.start(), entity_B.end() - 4 - 5)
            if entity_span_A[0] < entity_span_B[0]:
                entity_span_B = (entity_span_B[0] - 4 - 5, entity_span_B[1] - 4 - 5)
            else:
                entity_span_A = (entity_span_A[0] - 4 - 5, entity_span_A[1] - 4 - 5)
            # remove markers from text
            sent = re.sub(r'<(/?)e1>', '', sent)
            sent = re.sub(r'<(/?)e2>', '', sent)
            # get label
            label = relation_line.strip()

            # yield features
            yield RelExDatasetItem(
                sentence=sent,
                source_entity_span=entity_span_A,
                target_entity_span=entity_span_B,
                relation_type=SemEval2010Task8.LABELS.index(label)
            )