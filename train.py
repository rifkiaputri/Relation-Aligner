import dataset as dt
import siamese
import spacy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


nlp = spacy.load('en_core_web_sm')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def get_token(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        append(token.text)
    return tokens


# Todo: get word embedding ID given token
def get_embed_id(word):
    return 0


def get_padded_tensor(texts):
    text_t = [torch.tensor([get_embed_id(w) for w in get_token(text)], dtype=torch.long, device=device) for text in texts]
    text_l = [a.shape[0] for a in text_t]
    text_max = max(text_l)
    text_p = [text_max - a for a in text_l]
    text_t = [F.pad(a.view(1,1,1,-1), (0, text_p[i], 0, 0)).view(1,-1) for i, a in enumerate(text_t)]
    return torch.cat(text_t, 0)

    
def build_tensor(rels_kb, rels_oie, labels):
    label_t = torch.tensor(labels, dtype=torch.long, device=device)
    rel_kb_t = get_padded_tensor(rels_kb)
    rel_oie_t = get_padded_tensor(rels_oie)
    return rel_kb_t, rel_oie_t, label_t


def eval(loader, model):
    model.eval()
    corrects, avg_loss, size = 0, 0, 0
    c_loss = siamese.ContrastiveLoss()
    
    for rel_kb, rel_oie, label in loader:
        rel_kb_t, rel_oie_t, label_t = build_tensor(rel_kb, rel_oie, label)
        logit = model(rel_kb_t, rel_oie_t)
        loss = c_loss(logit, label_t)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(label_t.size()).data == label_t.data).sum()
        size += len(label)
        
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, size))
    return accuracy


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
        for rel_kb, rel_oie, label in train_loader:
            rel_kb_t, rel_oie_t, label_t = build_tensor(rel_kb, rel_oie, label)
            optimizer.zero_grad()
            logit = model(rel_kb_t, rel_oie_t)
            loss = c_loss(logit, label_t)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(label_t.size()).data == label_t.data).sum()
                accuracy = 100.0 * corrects / label_t.shape[0]
                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             label_t.shape[0]))
        # Eval validation data
        dev_acc = eval(valid_loader, model)
        
        if epoch % args.save_interval == 0:
            save(model, args.save_dir, 'model_temp', steps)

        
def main():
    # Load parameters
    BATCH_SIZE = 128
    
    class args:
        pass

    args = args()
    args.class_num = 2
    args.kernel_num = 32
    args.kernel_sizes = [3, 4, 5]
    args.dropout = 0.5
    args.static = True
    args.lr = 0.001
    args.epochs = 50
    args.log_interval = 10
    args.test_interval = 400
    args.save_interval = 10
    args.save_dir = 'models'  # model save path
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # Load train, valid, and test data
    train_dataset = dt.MyDataset('dataset/train_align.tsv')
    valid_dataset = dt.MyDataset('dataset/valid_align.tsv')
    test_dataset = dt.MyDataset('dataset/test_align.tsv')
    print('train, valid, test num:', len(train_dataset), len(valid_dataset), len(test_dataset))
    
    # Load dataset to DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = siamese.SiameseNetwork(args)
    model.double()
    model.to(device)
    
    # Train model
    try:
        train(model, train_loader, test_loader, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exit from training early')


if __name__ == '__main__':
    main()
