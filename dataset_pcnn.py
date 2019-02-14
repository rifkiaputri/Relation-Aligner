import csv
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, filename, mode='defent'):
        # Checking mode
        print('Mode:', mode)
        if mode not in ['nodef', 'defent']:
            raise ValueError('Invalid data mode')
        
        print('Load file', filename)
        data_header = [
            'e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
            'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id', 'oie_def', 'label'
        ]
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header, quoting=csv.QUOTE_NONE)
        
        self.labels = align_data['label']
        self.len = len(self.labels)
        e1_oie = align_data['e1_oie']
        e2_oie = align_data['e2_oie']
        rels_kb = align_data['rel_id']
        rels_oie = align_data['rel_oie']
        
        print('Load open IE relation & entity information')
        self.item_kb = []
        self.item_oie = []
        for i in range(self.len):
            e1_kb_item = align_data['e1_kb'][i]
            e2_kb_item = align_data['e2_kb'][i]
            e1_oie_item = align_data['e1_oie'][i]
            e2_oie_item = align_data['e2_oie'][i]
            
            if mode == 'defent':         
                # Load item information
                self.item_kb.append([e1_kb_item, align_data['rel_kb'][i] + ' sepunktoken ' + id2desc[rel_id], e2_kb_item])
                self.item_oie.append([e1_oie_item, rels_oie[i] + ' sepunktoken ' + align_data['oie_def'][i], e2_oie_item])
            else:
                self.item_kb.append([e1_kb_item, align_data['rel_kb'][i], e2_kb_item])
                self.item_oie.append([e1_oie_item, rels_oie[i], e2_oie_item])
            
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.item_kb[index], self.item_oie[index], self.labels[index]
