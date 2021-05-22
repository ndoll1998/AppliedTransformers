import os
from .base import AOEx_Dataset, AOEx_DatasetItem
from applied.common.path import FilePath

class SemEval2015Task12(AOEx_Dataset):
    """ SemEval 2015 Task 12 dataset for Aspect-Opinion Extraction
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """    

    # begin-in-out labeling scheme
    LABELS = ["O", "B", "I"]

    n_train_items = lambda self: 1316
    n_eval_items = lambda self: 686
    
    def yield_items(self, aspect_fname:str, sent_opinion_fname:str):
        # build full paths to files
        aspect_fpath = self.data_base_dir / aspect_fname
        sent_opinion_fpath = self.data_base_dir / sent_opinion_fname

        # load file contents
        with open(aspect_fpath, 'r', encoding='utf-8') as f:
            all_aspects = f.read().replace('NULL', '').split('\n')
        with open(sent_opinion_fpath, 'r', encoding='utf-8') as f:
            all_sents_opinions = f.read().split('\n')
        assert len(all_aspects) == len(all_sents_opinions)

        # preprocess data
        for sent_opinions, aspects in zip(all_sents_opinions, all_aspects):
            # separate sentence from opinions
            sent, opinions = sent_opinions.split('##') if '##' in sent_opinions else (sent_opinions, '')
            # get aspects and opinions
            opinions = [o.strip()[:-3] for o in opinions.split(',')] if len(opinions) > 0 else []
            aspects = [a.strip() for a in aspects.split(',')] if len(aspects) > 0 else []
            # build aspect and opinion spans
            opinion_pos = [sent.find(o) for o in opinions]
            opinion_spans = [(i, i + len(o)) for i, o in zip(opinion_pos, opinions)]
            aspect_pos = [sent.find(a) for a in aspects]
            aspect_spans = [(i, i + len(a)) for i, a in zip(aspect_pos, aspects)]
            # yield dataset item
            yield AOEx_DatasetItem(
                sentence=sent, 
                aspect_spans=aspect_spans, 
                opinion_spans=opinion_spans
            )

    # yield train and test items
    yield_train_items = lambda self: self.yield_items(
        aspect_fname=FilePath(
            "SemEval2015-Task12/aspectTerm_res15", 
            "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/aspectTerm_res15"
        ),
        sent_opinion_fname=FilePath(
            "SemEval2015-Task12/sentence_res15_op", 
            "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_res15_op"
        )
    )
    yield_eval_items = lambda self: self.yield_items(
        aspect_fname=FilePath(
            "SemEval2015-Task12/aspectTerm_restest15", 
            "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/aspectTerm_restest15"
        ),
        sent_opinion_fname=FilePath(
            "SemEval2015-Task12/sentence_restest15_op", 
            "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_restest15_op"
        )
    )
