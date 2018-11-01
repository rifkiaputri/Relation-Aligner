import pandas as pd
import csv
import string
import json
translator = str.maketrans('', '', string.punctuation)
import spacy
nlp = spacy.load('en_core_web_md')
import enteater
from os import listdir
from os.path import isfile, join


def get_noun_root(text, basic_root=True):
    doc = nlp(text)
    root = 'N/A'
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'ROOT':
            if basic_root == True:
                root = chunk.root.text
            else:
                root = chunk.text
    return root


def get_entity_tuple_id(e1, rel, e2):
    e1_wiki_id = 'N/A'
    e2_wiki_id = 'N/A'
    e1_label = 'N/A'
    e2_label = 'N/A'
    e1_root = get_noun_root(e1)
    if e1_root == 'N/A':
        e1_root = e1
    e2_root = get_noun_root(e2, basic_root=False)
    if e2_root == 'N/A':
        e2_root = e2
    sent = e1_root + ' ' + rel + ' ' + e2_root
    
    ent_result, ent_id_dict = enteater.get_entity(sent, detect_property=False)
    ent_rank = [ent[0] for ent in ent_result]
    for rel_item in rel.split():
        if rel_item in ent_rank:
            ent_rank.remove(rel_item)
    
    try:
        e1_idx = ent_rank.index(e1_root)
        # First entity is exist in wikidata
        e1_wiki_id = ent_id_dict[e1_root]
        if len(ent_rank) > 1:
            e1_idx = ent_rank.index(e1_root)
            if e1_idx == 0:
                e2_wiki_id = ent_id_dict[ent_rank[1]]
            else:
                e2_wiki_id = ent_id_dict[ent_rank[0]]
    except ValueError:
        # First entity is not exist in wikidata
        e1_wiki_id = 'N/A'
        if len(ent_rank) > 1:
            e2_wiki_id = ent_id_dict[ent_rank[0]]
    
    return (e1_wiki_id, e1_root, e2_wiki_id, e2_root)


def normalize_str(text):
    text = text.translate(translator)
    text.replace('LRB', ' ')
    return text.strip()

with open('../graphene/triples/dictionary/dict_ent.json') as f:
    dict_ent = json.loads(f.read())
dict_ent = dict((v,k) for k,v in dict_ent.items())
with open('../graphene/triples/dictionary/dict_rel.json') as f:
    dict_rel = json.loads(f.read())
dict_rel = dict((v,k) for k,v in dict_rel.items())
kb_rels = pd.read_csv('../relation_desc_triples.tsv', sep='\t', header=None, names=['id', 'label', 'desc', 'e1_id', 'e2_id', 'e1_label', 'e2_label'])


mypath = '../graphene/triples/cg-evidence/'
file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
len_file = len(file_list)

for i in range(len_file):
    filename = file_list[i]
    print(str(i) + '/' + str(len_file) + ': ' + filename)
    triples = pd.read_json('../graphene/triples/cg-evidence/' + filename, orient='records')
    align_data = []
    len_openie = len(triples)
    len_kb_rels = len(kb_rels)
    for idx in range(len_openie):
        triple_list = triples['Triples'][idx]
        if len(triple_list) > 0:
            for item in triple_list:
                arg1 = dict_ent[item[0]]
                rel = dict_rel[item[1]]
                arg2 = dict_ent[item[2]]
                valid = not isinstance(arg1, float) and not isinstance(arg2, float) and not isinstance(rel, float) and arg1.strip() and arg2.strip() and rel.strip()
                if valid:
                    e1_oie = normalize_str(arg1)
                    e2_oie = normalize_str(arg2)
                    rel_oie = normalize_str(rel)
                    if e1_oie and e2_oie and rel_oie:
                        e1_oie_id, e1_oie_root, e2_oie_id, e2_oie_root = get_entity_tuple_id(e1_oie, rel_oie, e2_oie)
                        for i in range(len_kb_rels):
                            rel_kb = kb_rels['label'][i]
                            rel_kb_id = kb_rels['id'][i]
                            if e1_oie_id.startswith('Q') and e2_oie_id.startswith('Q'):
                                align_data.append((kb_rels['e1_label'][i], rel_kb, rel_kb_id, kb_rels['e2_label'][i],
                                                   e1_oie, rel_oie, e2_oie,
                                                   kb_rels['e1_id'][i], kb_rels['e2_id'][i],
                                                   e1_oie_id, e2_oie_id, e1_oie_root, e2_oie_root,
                                                   '1'))

    input_filename = '../graphene/triples/cg-evidence-input/' + filename[:-5] + '.tsv'
    with open(input_filename, 'w') as f:
        for item in align_data:
            len_item = len(item) - 1
            for i in range(len_item):
                f.write(str(item[i]) + '\t')
            f.write(str(item[len_item]) + '\n')
    f.closed

    data_header = ['e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
                   'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id',
                   'e1_oie_root', 'e2_oie_root', 'label']
    data = pd.read_csv(input_filename, sep='\t', header=None, names=data_header, quoting=csv.QUOTE_NONE)
    data = data.drop_duplicates(keep=False)
    data.to_csv(input_filename, sep='\t', header=False, index=False)
    print(input_filename + ' saved')
