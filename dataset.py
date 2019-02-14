import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader


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
    '''
    Mode: nodef   -> no definition (only use relation phrase)
          def     -> with definition (use relational definition)
          defent  -> definition + entity information
    '''
    def __init__(self, filename, mode='defent'):
        # Checking mode
        print('Mode:', mode)
        if mode not in ['nodef', 'def', 'defent']:
            raise ValueError('Invalid data mode')
        
        # File loading
        print('Load file', filename)
        data_header = [
            'e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
            'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id', 'oie_def', 'label'
        ]
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header)
        
        rels_kb = align_data['rel_id']
        rels_oie = align_data['rel_oie']
        self.labels = align_data['label']
        self.len = len(self.labels)
        
        self.item_kb = []
        self.item_oie = []
        for i in range(self.len):
            if mode == 'defent':
                e1_kb = align_data['e1_kb'][i]
                e2_kb = align_data['e2_kb'][i]
                e1_oie = align_data['e1_oie'][i]
                e2_oie = align_data['e2_oie'][i]
                self.item_kb.append(e1_kb + ' ' + align_data['rel_kb'][i] + ' sepunktoken ' + id2desc[rels_kb[i]] + ' ' + e2_kb)
                self.item_oie.append(e1_oie + ' ' + rels_oie[i] + ' sepunktoken ' + align_data['oie_def'][i] + ' ' + e2_oie)
            elif mode == 'def':
                self.item_kb.append(align_data['rel_kb'][i] + ' sepunktoken ' + id2desc[rels_kb[i]])
                self.item_oie.append(rels_oie[i] + ' sepunktoken ' + align_data['oie_def'][i])
            elif mode == 'nodef':
                self.item_kb.append(align_data['rel_kb'][i])
                self.item_oie.append(rels_oie[i])
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.item_kb[index], self.item_oie[index], self.labels[index]
