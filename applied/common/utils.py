
def fetch(url, save_to=None):
    import requests, os, hashlib, tempfile
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
    print("fetching %s" % url)
    dat = requests.get(url).content
    with open(fp+".tmp", "wb") as f:
        f.write(dat)
    os.rename(fp+".tmp", save_to)
    return dat