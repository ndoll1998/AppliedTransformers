import os
import xml.etree.ElementTree as ET
from .base import NEC_Dataset, NEC_DatasetItem
from applied.common.path import FilePath
from applied.common.dataset import XML_Dataset

class SemEval2015Task12_AspectPolarity(XML_Dataset, NEC_Dataset):
    """ Dataset for the SemEval2014 Task4 data for Aspect-based Sentiment Analysis
        Download: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
    """

    LABELS = ['positive', 'neutral', 'negative']
    # train and eval files
    TRAIN_FILE = FilePath(
        "SemEval2015-Task12/ABSA-15_Restaurants_Train_Final.xml", 
        "https://raw.githubusercontent.com/peace195/aspect-based-sentiment-analysis/master/data/ABSA_SemEval2015/Restaurants_Train_Final.xml"
        )
    EVAL_FILE = FilePath(
        "SemEval2015-Task12/ABSA15_Restaurants_Test.xml", 
        "https://raw.githubusercontent.com/peace195/aspect-based-sentiment-analysis/master/data/ABSA_SemEval2015/Restaurants_Test.xml"
    )

    n_train_items = lambda self: 1315
    n_eval_items = lambda self: 685

    XSL_TEMPLATE = """
        <?xml version="1.0"?>
        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

            <!-- ignore root element since each item/sentence is processed individually -->
            <xsl:template match="Review"/>
            <xsl:template match="Reviews"/>
            <xsl:template match="sentences"/>

            <!-- collect only valuable information about a singel annotated sentence -->
            <xsl:template match="sentence">
                <root>
                    <sentence> <xsl:value-of select="text"/> </sentence>

                    <!-- collect all aspect spans -->
                    <entity_spans type="list">
                        <xsl:for-each select="Opinions/Opinion">
                        <item type="tuple">
                            <item type="int"><xsl:value-of select="@from"/></item>
                            <item type="int"><xsl:value-of select="@to"/>  </item>
                        </item>
                        </xsl:for-each>
                    </entity_spans>
                    
                    <!-- collect the corresponding labels -->
                    <labels type="list">
                        <xsl:for-each select="Opinions/Opinion">
                        <item><xsl:value-of select="@polarity"/></item>
                        </xsl:for-each>
                    </labels>

                </root>
            </xsl:template>

        </xsl:stylesheet>
    """
    
    def __init__(self, *args, **kwargs):
        # initialize dataset
        NEC_Dataset.__init__(self, *args, **kwargs)
        XML_Dataset.__init__(self,
            ItemType=NEC_DatasetItem,
            template=self.__class__.XSL_TEMPLATE,
            train_path=self.__class__.TRAIN_FILE,
            eval_path=self.__class__.EVAL_FILE
        )


class SemEval2015Task12_OpinionPolarity(NEC_Dataset):
    """ Dataset for the SemEval2014 Task4 data for Opinion-based Sentiment Analysis
        Downlaod: https://github.com/happywwy/Coupled-Multi-layer-Attentions/tree/master/util/data_semEval
    """

    LABELS = ['positive', 'negative']
    TRAIN_FILE = FilePath(
        "SemEval2015-Task12/sentence_res15_op", 
        "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_res15_op"
    )
    EVAL_FILE = FilePath(
        "SemEval2015-Task12/sentence_restest15_op", 
        "https://raw.githubusercontent.com/happywwy/Coupled-Multi-layer-Attentions/master/util/data_semEval/sentence_restest15_op"
    )

    n_train_items = lambda self: 760
    n_eval_items = lambda self: 333
    
    def yield_items(self, fpath:str) -> iter:
        # load file content
        with open(fpath, 'r', encoding='utf-8') as f:
            all_sents_opinions = f.read().split('\n')
        # preprocess data
        for sent_opinions in all_sents_opinions:
            # no opinions
            if '##' not in sent_opinions:
                continue
            # separate sentence from opinions
            sent, opinions = sent_opinions.split('##')
            # get aspects and opinions
            opinions = [o.strip() for o in opinions.split(',')] if len(opinions) > 0 else []
            opinions, sentiments = zip(*[(o[:-2].strip(), o[-2:]) for o in opinions])
            # build opinion spans
            opinion_pos = [sent.find(o) for o in opinions]
            opinion_spans = [(i, i + len(o)) for i, o in zip(opinion_pos, opinions)]
            # get sentiment labels
            sentiments = [self.__class__.LABELS[(1 - int(i)) // 2] for i in sentiments]
            # build dataset item
            yield NEC_DatasetItem(
                sentence=sent, 
                entity_spans=opinion_spans,
                labels=sentiments, 
            )
    
    # yield training and evaluation items
    yield_train_items = lambda self: self.yield_items(self.data_base_dir / self.__class__.TRAIN_FILE)
    yield_eval_items = lambda self: self.yield_items(self.data_base_dir / self.__class__.EVAL_FILE)

