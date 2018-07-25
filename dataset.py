import pandas as pd
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
    def __init__(self, filename):
        print('Load', filename)
        data_header = ['e1_kb', 'rel_kb', 'e2_kb', 'e1_openie', 'rel_openie', 'e2_openie', 'rel_id', 'label']
        align_data = pd.read_csv(filename, sep='\t', header=None, names=data_header)
            
        self.rels_kb = align_data['rel_id']
        self.rels_openie = align_data['rel_openie']
        self.labels = align_data['label']
        self.len = len(self.labels)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return id2desc[self.rels_kb[index]], self.rels_openie[index], self.labels[index]
