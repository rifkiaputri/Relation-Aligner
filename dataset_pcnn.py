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
    id2desc_len[rel_id] = rel_desc_len
print('Relation description loaded')


class MyDataset(Dataset):
    def __init__(self, filename):
        print('Load file', filename)
        data_header = ['e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
                       'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id',
                       'e1_oie_root', 'e2_oie_root', 'label']
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header, quoting=csv.QUOTE_NONE)
        
        self.rels_kb = align_data['rel_id']
        self.e1_kb = align_data['e1_kb']
        self.e2_kb = align_data['e2_kb']
        self.rels_oie = align_data['rel_oie']
        self.labels = align_data['label']
        self.len = len(self.labels)
        
        print('Load open IE relation definition & position vector')
        self.rels_oie_def = []
        self.e1_oie = []
        self.e2_oie = []
        self.positions = []
        self.item_kb = []
        self.item_oie = []
        for i in range(self.len):
            rel = str(align_data['rel_oie'][i])
            sent = str(align_data['e1_oie'][i]) + ' ' + rel + ' ' + str(align_data['e2_oie'][i])
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
            self.rels_oie_def.append(def_sent)
            
            # Change oie entities to root information
            e1_oie_root = str(align_data['e1_oie_root'][i])
            if e1_oie_root == 'N/A' or e1_oie_root is None or e1_oie_root == '':
                e1_oie_root = str(align_data['e1_oie'][i])
            e2_oie_root = str(align_data['e2_oie_root'][i])
            if e2_oie_root == 'N/A' or e2_oie_root is None or e1_oie_root == '':
                e2_oie_root = str(align_data['e2_oie'][i])
            self.e1_oie.append(str(e1_oie_root))
            self.e2_oie.append(str(e2_oie_root))
            
            # load position information
            e1_kb_tokens = str(self.e1_kb[i]).split()
            e1_kb_pos = len(e1_kb_tokens)
            e2_kb_pos = e1_kb_pos + int(id2desc_len[self.rels_kb[i]])
            if e2_kb_pos == e1_kb_pos:
                e2_kb_pos += 1

            e1_oie_tokens = str(self.e1_oie[i]).split()
            e1_oie_pos = len(e1_oie_tokens)
            e2_oie_tokens = tokenizer(def_sent)
            e2_oie_pos = len(e2_oie_tokens) + e1_oie_pos
            if e2_oie_pos == e1_oie_pos:
                e2_oie_pos += 1

            self.positions.append((e1_kb_pos, e2_kb_pos, e1_oie_pos, e2_oie_pos))
            
            # Load item information
            self.item_kb.append(str(self.e1_kb[i]) + ' ' + id2desc[self.rels_kb[i]] + ' ' + str(self.e2_kb[i]))
            self.item_oie.append(str(self.e1_oie[i]) + ' ' + self.rels_oie_def[i] + ' ' + str(self.e2_oie[i]))
            
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.item_kb[index], self.item_oie[index], self.positions[index], self.labels[index]
