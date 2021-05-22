from applied.core.dataset import Dataset, DatasetItem
from dataclasses import is_dataclass, fields
import numpy as np
import lxml.etree as ET

def elem2dict(node):
    """ Convert an lxml.etree node tree into a dict.
        Source: https://gist.github.com/jacobian/795571
    """
    result = {}
    for element in node.iterchildren():
        # Remove namespace prefix
        key = element.tag.split('}')[1] if '}' in element.tag else element.tag
        # Process element as tree element if the inner XML contains non-whitespace content
        if element.text and element.text.strip():
            value = element.text
        else:
            value = elem2dict(element)
        if key in result:            
            if type(result[key]) is list:
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value
    return result

class XML_Dataset(Dataset):
    """ Dataset reading from xml file and applying XSL
        Template to convert to dataset items
    """
    def __init__(self, ItemType:type, template:str, train_path:str, eval_path:str):
        # save dataset item type to use
        self.__ItemType = ItemType
        assert is_dataclass(ItemType)
        # save paths
        self.__train_path = self.data_base_dir / train_path
        self.__eval_path = self.data_base_dir / eval_path
        # build xstl-template
        template = ET.fromstring(template.strip())
        self.__xsl_template = ET.XSLT(template)

    def prepare_item_kwargs(self, kwargs:dict) -> dict:
        """ Keyword arguments preparation hook allowing for manual manipulation
            of the arguments passed to the item initializer. By default does nothing.
        """
        return kwargs

    def create_item(self, xml) -> DatasetItem:
        # convert xml to keyword arguments for item
        kwargs = elem2dict(xml)
        # make sure all items are present
        for f in fields(self.__ItemType):
            if f.name not in kwargs:
                return None
        # prepare keyword arguments
        kwargs = self.prepare_item_kwargs(kwargs)
        # create the dataset item
        return self.__ItemType(**kwargs)
 
    def parse_xml(self, fpath) -> iter:
        # read xml file
        with open(fpath, 'rb') as f:
            # iterate over tree and parse each item
            for _, elem in ET.iterparse(f):
                # transform using
                xml_items = self.__xsl_template(elem)
                # convert xml to dicts
                root = xml_items.getroot()
                if root is not None:
                    yield self.create_item(root)
    
    yield_train_items = lambda self: self.parse_xml(self.__train_path)    
    yield_eval_items = lambda self: self.parse_xml(self.__eval_path)    
