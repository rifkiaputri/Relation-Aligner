import moses
import pandas as pd
from nltk.wsd import lesk
from torch.utils.data import Dataset, DataLoader


tokenizer = moses.MosesTokenizer()

# Build dataset dictionary
print('Load relation description dictionary...')
id2desc = {}
desc_data = pd.read_csv('dataset/relation_desc.tsv', sep='\t', header=None, names=['id', 'label', 'desc'])
for i in range(len(desc_data['desc'])):
    rel_desc = desc_data['desc'][i]
    rel_id = desc_data['id'][i]
    id2desc[rel_id] = rel_desc
print('Relation description loaded')


class MyDataset(Dataset):
    def __init__(self, filename):
        print('Load file', filename)
        data_header = ['e1_kb', 'rel_kb', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie', 'rel_id', 'label',
                       'e1_oie_root', 'e2_oie_root', 'e1_desc', 'e2_desc']
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header)
        
        self.rels_kb = align_data['rel_id']
        self.e1_kb = align_data['e1_kb']
        self.e2_kb = align_data['e2_kb']
        self.rels_oie = align_data['rel_oie']
        self.e1_oie = align_data['e1_oie']
        self.e2_oie = align_data['e2_oie']
        self.labels = align_data['label']
        self.len = len(self.labels)
        
        print('Load open IE relation definition')
        self.rels_oie_def = []
        for i in range(self.len):
            rel = align_data['rel_oie'][i]
            sent = align_data['e1_oie'][i] + ' ' + rel + ' ' + align_data['e2_oie'][i]
            rel_tokens = tokenizer.tokenize(rel)
            sent_tokens = tokenizer.tokenize(sent)
            def_sent = rel
            for token in rel_tokens:
                syns = lesk(sent_tokens, token)
                if syns is not None:
                    if def_sent == rel:
                        def_sent = syns.definition()
                    else:
                        def_sent = def_sent + ' ' + syns.definition()
            self.rels_oie_def.append(def_sent)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item_kb = self.e1_kb[index] + ' ' + id2desc[self.rels_kb[index]] + ' ' + self.e2_kb[index]
        item_oie = self.e1_oie[index] + ' ' + self.rels_oie_def[index] + ' ' + self.e2_oie[index]
        return item_kb, item_oie, self.labels[index]
