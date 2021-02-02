import os
import requests
import tempfile
import hashlib

class FilePath(os.PathLike):
    """ Helper class for downloadable path """

    def __init__(self, 
        path:str, 
        url:str,
        post_fetch =lambda raw: raw
    ) -> None:
        # initialize path
        self.__path = path
        # function to apply after fetch
        self.__url = url
        self.__post_fetch = post_fetch

    def fetch(self) -> None:
        # build paths
        tmp_fname = hashlib.md5(self.__url.encode('utf-8')).hexdigest()
        tmp_fpath = os.path.join(tempfile.gettempdir(), tmp_fname)
        # fetch and postprocess
        raw = requests.get(self.__url).content
        raw = self.__post_fetch(raw)
        # write to temporary file
        with open("%s.tmp" % tmp_fpath, "wb") as f: 
            f.write(raw)
        # move to actual file path
        directory, _ = os.path.split(self.__path)
        os.makedirs(directory, exist_ok=True)
        os.rename("%s.tmp" % tmp_fpath, self.__path)

    def __fspath__(self):
        # fetch data
        if not os.path.isfile(self.__path):
            self.fetch()
            assert os.path.isfile(self.__path)
        # return file system path
        return self.__path

    def __str__(self) -> str:
        return self.__path

    def __rtruediv__(self, other):
        assert not isinstance(other, FilePath), "Cannot concatenate two fetchable paths"
        return FilePath(
            path=os.path.join(other, self.__path),
            url=self.__url,
            post_fetch=self.__post_fetch
        )
