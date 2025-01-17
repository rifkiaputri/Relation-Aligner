import torch
import torch.nn as nn
import numpy as np
from gensim.models import FastText
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_vectors():
    '''
    create word embedding without reducing the dimension
    '''
    print('Load word2vec bin file...')
    word_vectors = FastText.load_fasttext_format('./dataset/wordvector/wiki.en')
    vec_list = [word_vectors[i] for i in word_vectors.wv.vocab.keys()]
    
    print('Writing vectors to file...')
    vec_file = open('./dataset/wordvector/vec_300.txt', 'w')
    for word, vec in zip(word_vectors.wv.vocab.keys(), vec_list):
        vec_str = ' '.join(str(x) for x in vec)
        vec_file.write(vec_str + '\n')
    vec_file.close()
    print('Successfully saved vectors in ./dataset/wordvector directory')
    

def create_pca_vectors(pc_number = 50):
    print('Load word2vec bin file...')
    word_vectors = FastText.load_fasttext_format('./dataset/wordvector/wiki.en')
    vec_list = [word_vectors[i] for i in word_vectors.wv.vocab.keys()]
    
    print('Normalize vector...')
    normalized_vec = normalize(vec_list)
    
    print('Performing PCA...')
    pca = PCA(n_components=pc_number)
    pca.fit(normalized_vec)
    reduced_vec = pca.transform(normalized_vec)
    
    print('Writing PCA vectors to file...')
    word_to_id_file = open('./dataset/wordvector/word_to_id.txt', 'w')
    vec_file = open('./dataset/wordvector/vec_' + str(pc_number) + '.txt', 'w')
    for word, vec in zip(word_vectors.wv.vocab.keys(), reduced_vec):
        word_to_id_file.write(word + '\n')
        vec_str = ' '.join(str(x) for x in vec)
        vec_file.write(vec_str + '\n')
    word_to_id_file.close()
    vec_file.close()
    print('Successfully saved vectors in ./dataset/wordvector directory')

    
def get_word_vectors(vec_file, dim):
    print('Read vector file...')
    with open(vec_file) as f:
        vec_string = f.read().splitlines()
    
    print('Initialize word vector array...')
    # Note: index 0 will be initialized with 0 vector
    wv = np.zeros((len(vec_string) + 1, dim), dtype=np.float32)
    i = 1
    for vec in vec_string:
        wv[i] = np.fromstring(vec, dtype=np.float32, sep=' ')
        i += 1
    
    print('Convert word vector to tensor...')
    wv = torch.from_numpy(wv)
    
    if torch.cuda.is_available():
        wv = wv.cuda()
    
    return wv


def get_embedding(vec_file='./dataset/wordvector/vec_50.txt', dim=50):
    '''
    return: Initialized torch embedding, vocabulary number, word embedding dimension
    '''
    word_vector = get_word_vectors(vec_file, dim)
    vocab_size, vec_size = word_vector.size(0), word_vector.size(1)
    embed = nn.Embedding.from_pretrained(word_vector)
    return (embed, vocab_size, vec_size)


def get_embedding_300():
    print('Load fastText .vec file...')
    word_vectors = KeyedVectors.load_word2vec_format('dataset/wordvector/wiki-subword-300d.vec')
    weights = torch.from_numpy(np.pad(word_vectors.wv.vectors, [(1,0),(0,0)], mode='constant')).to(device)
    embed = nn.Embedding.from_pretrained(weights)
    return (embed, 300)