from .base import ABSA_Dataset, ABSA_DatasetItem
from applied.common.dataset import XML_Dataset
from applied.common.path import FilePath

class __SemEval2014Task4(ABSA_Dataset, XML_Dataset):

    # labels
    LABELS = ['positive', 'neutral', 'negative', 'conflict']
    # train and eval data files
    TRAIN_FILE = None
    EVAL_FILE = None

    # define xls-template    
    XSL_TEMPLATE = """
        <?xml version="1.0"?>
        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

            <!-- ignore root element since each item/sentence is processed individually -->
            <xsl:template match="sentences"/>

            <!-- collect only valuable information about a singel annotated sentence -->
            <xsl:template match="sentence">
                <root>
                    <sentence> <xsl:value-of select="text"/> </sentence>

                    <!-- aspect terms with labels -->
                    <xsl:for-each select="aspectTerms/aspectTerm">
                    <aspects>
                        <xsl:value-of select="@term"/>
                    </aspects>
                    <labels>
                        <xsl:value-of select="@polarity"/>
                    </labels>
                    </xsl:for-each>

                    <!-- aspect categories with labels -->
                    <xsl:for-each select="aspectCategories/aspectCategory">
                    <aspects>
                        <xsl:value-of select="@category"/>
                    </aspects>
                    <labels>
                        <xsl:value-of select="@polarity"/>
                    </labels>
                    </xsl:for-each>
                </root>
            </xsl:template>

        </xsl:stylesheet>
        """

    def prepare_item_kwargs(self, kwargs:dict) -> dict:
        # convert labels from string to index
        kwargs['labels'] = [kwargs['labels']] if isinstance(kwargs['labels'], str) else kwargs['labels']
        kwargs['labels'] = [SemEval2014Task4_Restaurants.LABELS.index(l) for l in kwargs['labels']]
        return kwargs

    def __init__(self, *args, **kwargs):
        # initialize dataset
        ABSA_Dataset.__init__(self, *args, **kwargs)
        XML_Dataset.__init__(self,
            ItemType=ABSA_DatasetItem,
            template=SemEval2014Task4_Restaurants.XSL_TEMPLATE,
            train_path=SemEval2014Task4_Restaurants.TRAIN_FILE,
            eval_path=SemEval2014Task4_Restaurants.EVAL_FILE
        )


class SemEval2014Task4_Restaurants(__SemEval2014Task4):
    """ SemEval 2014 Task 4 Restaurant Dataset for Aspect based Sentiment Analysis.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    
    # paths
    TRAIN_FILE = FilePath(
        "SemEval2014-Task4/Restaurants_Train.xml",
        "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Restaurants_Train_v2.xml"
    )
    EVAL_FILE = FilePath(
        "SemEval2014-Task4/restaurants-trial.xml",
        "https://alt.qcri.org/semeval2014/task4/data/uploads/restaurants-trial.xml"
    )
    
    n_train_items = lambda self: 3041
    n_eval_items = lambda self: 100
   

class SemEval2014Task4_Laptops(__SemEval2014Task4):
    """ SemEval 2014 Task 4 Laptop Dataset for Aspect based Sentiment Analysis.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """

    # paths
    TRAIN_FILE = FilePath(
        "SemEval2014-Task4/Laptops_Train.xml",
        "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Laptop_Train_v2.xml"
    )
    TEST_FILE = FilePath(
        "SemEval2014-Task4/laptops-trial.xml",
        "https://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml"
    )

    n_train_items = lambda self: 3045
    n_eval_items = lambda self: 100


class SemEval2014Task4(SemEval2014Task4_Restaurants, SemEval2014Task4_Laptops):
    """ SemEval 2014 Task 4 Dataset for Aspect based Sentiment Analysis.
        Combination of the restaurant and laptop dataset.
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """

    n_train_items = lambda self: 6082
    n_eval_items = lambda self: 200

    def yield_train_items(self) -> iter:
        # yield from restaurant and from laptop dataset
        yield from SemEval2014Task4_Restaurants.yield_train_items(self)
        yield from SemEval2014Task4_Laptops.yield_train_items(self)
    def yield_eval_items(self) -> iter:
        # yield from restaurant and from laptop dataset
        yield from SemEval2014Task4_Restaurants.yield_eval_items(self)
        yield from SemEval2014Task4_Laptops.yield_eval_items(self)


class SemEval2014Task4_Category(__SemEval2014Task4):
    """ SemEval 2014 Task 4 Aspect-Category Dataset for Aspect based Sentiment Analysis.
        Only provides examples for aspect-categories (not explicitly mentioned in the text).
        Download: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
    """
    
    # paths
    TRAIN_FILE = FilePath(
        "SemEval2014-Task4/Restaurants_Train.xml",
        "https://raw.githubusercontent.com/pedrobalage/SemevalAspectBasedSentimentAnalysis/master/semeval_data/Restaurants_Train_v2.xml"
    )
    TEST_FILE = FilePath(
        "SemEval2014-Task4/restaurants-trial.xml",
        "https://alt.qcri.org/semeval2014/task4/data/uploads/restaurants-trial.xml"
    )
    
    n_train_items = lambda self: 3041
    n_eval_items = lambda self: 100
    
    # define xls-template    
    XSL_TEMPLATE = """
        <?xml version="1.0"?>
        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

            <!-- ignore root element since each item/sentence is processed individually -->
            <xsl:template match="sentences"/>

            <!-- collect only valuable information about a singel annotated sentence -->
            <xsl:template match="sentence">
                <root>
                    <sentence> <xsl:value-of select="text"/> </sentence>

                    <!-- aspect categories with labels -->
                    <xsl:for-each select="aspectCategories/aspectCategory">
                    <aspects>
                        <xsl:value-of select="@category"/>
                    </aspects>
                    <labels>
                        <xsl:value-of select="@polarity"/>
                    </labels>
                    </xsl:for-each>

                </root>
            </xsl:template>

        </xsl:stylesheet>
        """
