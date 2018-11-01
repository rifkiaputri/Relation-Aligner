import dataset_pcnn as dt
import os
import re
import siamese_pcnn as siamese
import torch
import torch.nn.functional as F
import wordvector as wv
import pandas as pd
import params
from datetime import datetime
from torch.utils.data import DataLoader
from mosestokenizer import MosesTokenizer

# Initialization
 
tokenizer = MosesTokenizer('en')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Load word dictionary
word_dic = {}
with open(os.path.join('dataset', 'wordvector', 'word_to_id.txt'), 'r', encoding='utf-8') as f:
    i = 1
    for line in f:
        word = line.strip()
        word_dic[word] = i
        i += 1
        

def get_token(text):
    text = ' '.join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
    tokens = tokenizer(text)
    return tokens


def get_embed_id(word):
    return word_dic.get(word, 0)


def get_padded_tensor(texts):
    text_t = [torch.tensor([get_embed_id(w) for w in get_token(text)], dtype=torch.long, device=device) for text in texts]
    text_l = [a.shape[0] for a in text_t]
    text_max = max(text_l)
    text_p = [text_max - a for a in text_l]
    text_t = [F.pad(a.view(1,1,1,-1), (0, text_p[i], 0, 0)).view(1,-1) for i, a in enumerate(text_t)]
    text_t = torch.cat(text_t, 0)
    return text_t

    
def build_tensor(rels_kb, rels_oie, labels):
    label_t = torch.tensor(labels, dtype=torch.float32, device=device)
    rel_kb_t = get_padded_tensor(rels_kb)
    rel_oie_t = get_padded_tensor(rels_oie)
    return rel_kb_t, rel_oie_t, label_t


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    if steps > 0:
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    else:
        save_path = '{}.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
            
            
def test(test_loader, model, args):
    predict = []
    
    # restore the best parameters
    print('Load model parameters...')
    model_file = os.path.join(args.save_dir, args.model_filename + '.pt')
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    
    model.eval()
    corrects = 0
    
    print('Predict test data...')
    with torch.no_grad():
        for rel_kb, rel_oie, position, label in test_loader:
            rel_kb_t, rel_oie_t, label_t = build_tensor(rel_kb, rel_oie, label)
            output1, output2 = model(rel_kb_t, rel_oie_t, position)
            euclidean_dist = F.pairwise_distance(output1, output2)
            dist = euclidean_dist.cpu().numpy().tolist()
            for i in range(len(dist)):
                predict.append(dist[i])
    
    return predict

        
def main():
    # Load parameters
    BATCH_SIZE = 1000
    args = params.args()
    input_filename = 'Great_Britain'
    
    # Load train, valid, and test data
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start reading test data...')
    data_header = ['e1_kb', 'rel_kb', 'rel_id', 'e2_kb', 'e1_oie', 'rel_oie', 'e2_oie',
                   'e1_kb_id', 'e2_kb_id', 'e1_oie_id', 'e2_oie_id',
                   'e1_oie_root', 'e2_oie_root', 'label']
    test_align = pd.read_csv('dataset/graphene/triples/cg-evidence-input/' + input_filename + '.tsv', sep='\t', header=None, names=data_header, quoting=3)
    test_dataset = dt.MyDataset('dataset/graphene/triples/cg-evidence-input/' + input_filename + '.tsv')
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Test num:', len(test_dataset))
    
    # Load dataset to DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = siamese.SiameseNetwork(args)
    model.to(device)
    
    # Test model
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start predict alignment...')
    predict = test(test_loader, model, args)
    
    if not os.path.exists(args.predict_dir):
        os.mkdir(args.predict_dir)
    
    threshold = 2.6
    pred_filename = args.predict_dir + '/' + input_filename + '_predict_result.tsv'
    with open(pred_filename, 'w') as f:
        f.write('e1 oie\trel oie\te2 oie\te1 kb id\te2 kb id\trel kb id\trel kb\tdist\n')
        for i in range(len(predict)):
            if predict[i] < threshold:
                f.write(test_align['e1_oie'][i] + '\t' + test_align['rel_oie'][i] + '\t' + test_align['e2_oie'][i] + '\t')
                f.write(str(test_align['e1_oie_id'][i]) + '\t' + str(test_align['e2_oie_id'][i]) + '\t')
                f.write(test_align['rel_id'][i] + '\t' + test_align['rel_kb'][i] + '\t' + str(predict[i]))
                f.write('\n')
    f.closed
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Successfully save prediction result to', pred_filename)
            

if __name__ == '__main__':
    main()
