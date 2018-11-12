# Taken from https://gist.github.com/bstriner/7062dbefd54bd66955a4aa67f8f0cdc4

import glob
import os
import json


class WikiDoc(object):
    def __init__(self, url, text, id, title):
        self.url = url
        self.text = text
        self.id = id
        self.title = title


class WikiModel(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def files(self):
        return glob.glob(os.path.join(self.data_dir, "**", "wiki_*"))

    def file_docs(self, path):
        with open(path) as f:
            for line in f:
                if line:
                    doc = json.loads(line)
                    yield WikiDoc(doc["url"], doc["text"], doc["id"], doc["title"])

    def docs(self):
        for path in self.files():
            for doc in self.file_docs(path):
                yield doc