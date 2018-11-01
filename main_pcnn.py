import dataset_pcnn as dt
import moses
import os
import params
import re
import siamese_pcnn as siamese
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader


# Initialization
tokenizer = moses.MosesTokenizer()
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
    tokens = tokenizer.tokenize(text)
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
    
    
def eval(loader, model):
    model.eval()
    avg_loss, size = 0, 0
    c_loss = siamese.ContrastiveLoss()
    
    for rel_kb, rel_oie, position, label in loader:
        rel_kb_t, rel_oie_t, label_t = build_tensor(rel_kb, rel_oie, label)
        output1, output2 = model(rel_kb_t, rel_oie_t, position)
        loss = c_loss(output1, output2, label_t)
        avg_loss += loss.data[0]
        size += 1
        
    avg_loss = avg_loss / size
    print('Evaluation - Current loss {}'.format(avg_loss.data[0]))


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    if steps > 0:
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    else:
        save_path = '{}.pt'.format(save_prefix)
    torch.save(model.state_dict(), save_path)
    

def train(model, train_loader, valid_loader, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    model.train()
    
    steps = 0
    c_loss = siamese.ContrastiveLoss()
    
    for epoch in range(1, args.epochs + 1):
        print('\nStart epoch', epoch)
        for rel_kb, rel_oie, position, label in train_loader:
            rel_kb_t, rel_oie_t, label_t = build_tensor(rel_kb, rel_oie, label)
            optimizer.zero_grad()
            output1, output2 = model(rel_kb_t, rel_oie_t, position)
            loss = c_loss(output1, output2, label_t)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                print('Batch[{}] - Current loss: {:.6f}'.format(steps, loss.data[0]))
                
        # Eval validation data
        eval(valid_loader, model)
        
        if epoch % args.save_interval == 0:
            save(model, args.save_dir, 'model_temp', steps)
            
            
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
            gold = label_t.cpu().numpy().tolist()
            dist = euclidean_dist.cpu().numpy().tolist()
            for i in range(len(gold)):
                predict.append((rel_kb[i], rel_oie[i], dist[i], gold[i], output1[i], output2[i]))
    
    return predict

        
def main():
    # Load parameters
    args = params.args()
    
    # Load train, valid, and test data
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Loading dataset')
    train_dataset = dt.MyDataset(args.train_filename)
    valid_dataset = dt.MyDataset(args.valid_filename)
    test_dataset = dt.MyDataset(args.test_filename)
    print('train, valid, test num:', len(train_dataset), len(valid_dataset), len(test_dataset))
    
    # Load dataset to DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = siamese.SiameseNetwork(args)
    model.to(device)
    
    # Train model
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start training')
    try:
        train(model, train_loader, valid_loader, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exit from training early')
        
    # Save final model
    save(model, args.save_dir, args.model_filename, -1)
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Training finished')
    
    # Test model
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start prediction')
    predict = test(test_loader, model, args)
    
    if not os.path.exists(args.predict_dir):
        os.mkdir(args.predict_dir)
        
    pred_filename = args.predict_dir + '/predict_result.tsv'
    with open(pred_filename, 'w') as f:
        for item in predict:
            f.write(item[0] + '\t' + item[1] + '\t' + str(item[2]) + '\t' + str(item[3]) + '\n')
    f.closed
    print('Successfully save prediction result to', pred_filename)
    
    with open(args.predict_dir + '/rel_embed_vector.tsv', 'w') as f:
        for item in predict:
            out1 = item[5].cpu().numpy().tolist()
            f.write('\t'.join(str(x) for x in out1))
            f.write('\n')
    f.closed
    
    with open(args.predict_dir + '/rel_embed_label.tsv', 'w') as f:
        for item in predict:
            f.write(item[1])
            f.write('\n')
    f.closed
    print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Prediction finished')
            

if __name__ == '__main__':
    main()
