import csv
import pandas as pd
from nltk.wsd import lesk
from torch.utils.data import Dataset, DataLoader
from mosestokenizer import MosesTokenizer


tokenizer = MosesTokenizer('en')

# Build dataset dictionary
print('Load relation description dictionary...')
id2desc = {}
id2desc_len = {}
desc_data = pd.read_csv('dataset/relation_desc_len.tsv', sep='\t', header=None, names=['id', 'label', 'desc', 'desc_len'])
for i in range(len(desc_data['desc'])):
    rel_desc = desc_data['desc'][i]
    rel_desc_len = desc_data['desc_len'][i]
    rel_id = desc_data['id'][i]
    id2desc[rel_id] = rel_desc
    id2desc_len[rel_id] = int(rel_desc_len)
print('Relation description loaded')


class MyDataset(Dataset):
    def __init__(self, filename):
        print('Load file', filename)
        data_header = ['e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
                       'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id',
                       'e1_oie_root', 'e2_oie_root', 'label']
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header, quoting=csv.QUOTE_NONE)
        
        self.labels = align_data['label']
        self.len = len(self.labels)
        e1_oie = align_data['e1_oie']
        e2_oie = align_data['e2_oie']
        e1_kb = align_data['e1_kb']
        e2_kb = align_data['e2_kb']
        rels_kb = align_data['rel_id']
        rels_oie = align_data['rel_oie']
        
        print('Load open IE relation definition & position vector')
        self.item_kb = []
        self.item_oie = []
        for i in range(self.len):
            e1_oie_item = str(e1_oie[i])
            e2_oie_item = str(e2_oie[i])
            e1_kb_item = e1_kb[i]
            e2_kb_item = e2_kb[i]
            rel_id = rels_kb[i]
            rel = rels_oie[i]
            sent = e1_oie_item + ' ' + rel + ' ' + e2_oie_item
            
            rel_tokens = tokenizer(rel)
            sent_tokens = tokenizer(sent)
            def_sent = rel
            for token in rel_tokens:
                syns = lesk(sent_tokens, token)
                if syns is not None:
                    if def_sent == rel:
                        def_sent = syns.definition()
                    else:
                        def_sent = def_sent + ' ' + syns.definition()          
                
            # Load item information
            self.item_kb.append([e1_kb_item, id2desc[rel_id], e2_kb_item])
            self.item_oie.append([e1_oie_item, def_sent, e2_oie_item])
            
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.item_kb[index], self.item_oie[index], self.labels[index]
