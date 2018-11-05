import os
import wordvector as wv
from datetime import datetime


class args():
    def __init__(self):
        self.BATCH_SIZE = 128
        print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start loading word embedding...')
        self.embed, self.embeding_num, self.embeding_dim = wv.get_embedding(vec_file='./dataset/wordvector/vec_50.txt', dim=50)
        print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Word embedding loaded')
        self.output_num = 2
        self.dropout = 0.5
        self.static = True
        self.lr = 0.001
        self.epochs = 50
        self.log_interval = 20
        self.test_interval = 400
        self.save_interval = 10
        self.loss_log_interval = 50
        self.train_filename = 'dataset/train_align_filtered.tsv'
        self.valid_filename = 'dataset/valid_align_filtered.tsv'
        self.test_filename = 'dataset/test_align_filtered.tsv'
#         self.test_filename = 'dataset/test_align_gold.tsv'
        self.save_dir = 'models/data_v4'  # model save path
        
        # Mode
        self.mode = 'defent'
        self.predict_dir = 'predict/data_v4/pcnn_' + self.mode
        self.model_filename = 'pcnn_final_' + self.mode
        
        if self.mode == 'nodef':
            self.kernel_num = 8
            self.kernel_sizes = [1]
        else:
            self.kernel_num = 32
            self.kernel_sizes = [2, 3]
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
