# import base predictor
from core.Predictor import BasePredictor
# import base model and dataset type
from .models import AspectOpinionExtractionModel
from .datasets import AspectOpinionExtractionDataset
# import utils
from core.utils import build_token_spans, get_spans_from_bio_scheme


class AspectOpinionExtractionPredictor(BasePredictor):

    BASE_MODEL_TYPE = AspectOpinionExtractionModel
    BASE_DATASET_TYPE = AspectOpinionExtractionDataset

    def predict(self, text, *args, **kwargs):
        # predict
        aspect_token_spans, opinion_token_spans = BasePredictor.predict(self, text, *args, **kwargs)
        # build token spans
        tokens = self.tokenizer.tokenize(text)
        spans = build_token_spans(tokens, text)
        # get aspect and opinion terms
        aspect_terms = [text[spans[s][0]:spans[e-1][1]] for s, e in aspect_token_spans]
        opinion_terms = [text[spans[s][0]:spans[e-1][1]] for s, e in opinion_token_spans]
        
        # return
        return aspect_terms, opinion_terms

    def postprocess(self, aspect_logits, opinion_logits, *additionals):
        # get bio-schemes from logits
        aspect_bio = aspect_logits.max(dim=-1)[1].cpu().tolist()[0]
        opinion_bio = opinion_logits.max(dim=-1)[1].cpu().tolist()[0]
        # get token spans from bio scheme
        aspect_token_spans = get_spans_from_bio_scheme(aspect_bio)
        opinion_token_spans = get_spans_from_bio_scheme(opinion_bio)
        # return
        return aspect_token_spans, opinion_token_spans
